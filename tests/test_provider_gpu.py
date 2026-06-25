from vllmstat.providers.gpu import GpuProvider, parse_nvidia_smi_csv, read_nvml


class FakeNvml:
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2

    def nvmlInit(self): ...  # noqa: ANN201

    def nvmlShutdown(self): ...  # noqa: ANN201

    def nvmlDeviceGetCount(self):  # noqa: ANN201
        return 1

    def nvmlDeviceGetHandleByIndex(self, i):  # noqa: ANN201
        return i

    def nvmlDeviceGetName(self, h):  # noqa: ANN201
        return "NVIDIA Test GPU"

    def nvmlDeviceGetUtilizationRates(self, h):  # noqa: ANN201
        class U:
            gpu = 81
            memory = 60

        return U()

    def nvmlDeviceGetMemoryInfo(self, h):  # noqa: ANN201
        class M:
            used = 23_100_000_000
            total = 24_000_000_000

        return M()

    def nvmlDeviceGetTemperature(self, h, sensor):  # noqa: ANN201
        return 61

    def nvmlDeviceGetPowerUsage(self, h):  # noqa: ANN201
        return 142_000  # mW

    def nvmlDeviceGetEnforcedPowerLimit(self, h):  # noqa: ANN201
        return 200_000

    def nvmlDeviceGetFanSpeed(self, h):  # noqa: ANN201
        return 45

    def nvmlDeviceGetClockInfo(self, h, c):  # noqa: ANN201
        return 2520 if c == 1 else 9501


def test_read_nvml_builds_samples():
    snap = read_nvml(FakeNvml())
    assert snap.available and snap.source == "nvml"
    g = snap.gpus[0]
    assert g.name == "NVIDIA Test GPU"
    assert g.util_gpu == 81
    assert g.mem_used == 23_100_000_000
    assert g.power_w == 142.0
    assert g.clock_sm_mhz == 2520


def test_parse_nvidia_smi_csv():
    line = "0, NVIDIA Test GPU, 81, 23100, 24000, 61, 142.0, 200.0, 2520, 9501, 45"
    gpus = parse_nvidia_smi_csv(line)
    g = gpus[0]
    assert g.index == 0 and g.name == "NVIDIA Test GPU"
    assert g.mem_used == 23100 * 1024 * 1024  # MiB -> bytes
    assert g.power_w == 142.0


def test_provider_disabled_returns_unavailable():
    snap = GpuProvider(enabled=False).sample()
    assert snap.available is False and snap.source == "none"


def _intel_card(tmp_path):  # noqa: ANN001
    card = tmp_path / "card0"
    dev = card / "device"
    (dev).mkdir(parents=True)
    (dev / "vendor").write_text("0x8086\n")
    (dev / "device").write_text("0xe223\n")
    freq = dev / "tile0" / "gt0" / "freq0"
    freq.mkdir(parents=True)
    (freq / "cur_freq").write_text("2800\n")
    hw = dev / "hwmon" / "hwmon0"
    hw.mkdir(parents=True)
    (hw / "name").write_text("xe\n")
    (hw / "temp2_input").write_text("57000\n")
    (hw / "energy1_input").write_text("84023575462341\n")
    (hw / "power1_cap").write_text("275000000\n")
    (hw / "fan1_input").write_text("1060\n")
    return tmp_path


def test_provider_detects_intel_card_via_tmp_drm_root(tmp_path):  # noqa: ANN001
    drm = _intel_card(tmp_path)
    p = GpuProvider(enabled=True, drm_root=str(drm), clock=lambda: 100.0)
    snap = p.sample()
    assert snap.available is True
    assert "intel" in snap.source
    g = snap.gpus[0]
    assert g.vendor == "intel"
    assert g.clock_sm_mhz == 2800
    assert g.temp_c == 57.0
    assert g.power_limit_w == 275.0
    assert g.fan_rpm == 1060
    assert g.power_w is None  # first sample -> no power delta yet


def test_provider_intel_power_delta_across_two_samples(tmp_path):  # noqa: ANN001
    drm = _intel_card(tmp_path)
    hw = drm / "card0" / "device" / "hwmon" / "hwmon0"
    times = iter([100.0, 102.0])
    p = GpuProvider(enabled=True, drm_root=str(drm), clock=lambda: next(times))
    p.sample()  # primes the energy carry at t=100
    # advance the energy counter by 60 J (60_000_000 µJ) for the 2 s gap -> 30 W
    base = 84023575462341
    hw.joinpath("energy1_input").write_text(f"{base + 60_000_000}\n")
    snap = p.sample()
    power_w = snap.gpus[0].power_w
    assert power_w is not None
    assert abs(power_w - 30.0) < 1e-6


_FDINFO = """drm-driver:\txe
drm-pdev:\t{pdev}
drm-client-id:\t{cid}
drm-cycles-ccs:\t{ccs}
drm-total-cycles-ccs:\t{total}
drm-resident-vram0:\t{vram} KiB
"""


def _fdinfo_client(proc_root, pid, fd, *, pdev, cid, ccs, total, vram):  # noqa: ANN001
    d = proc_root / str(pid) / "fdinfo"
    d.mkdir(parents=True, exist_ok=True)
    (d / str(fd)).write_text(_FDINFO.format(pdev=pdev, cid=cid, ccs=ccs, total=total, vram=vram))


def test_provider_intel_picks_up_fdinfo_util_and_vram(tmp_path):  # noqa: ANN001
    # DRM card under one tmp tree; a fake /proc with GPU clients under another.
    drm = _intel_card(tmp_path / "sys")
    proc = tmp_path / "proc"
    pdev = "0000:06:00.0"
    _fdinfo_client(proc, 100, 3, pdev=pdev, cid=1, ccs=100, total=1000, vram=20_000_000)
    _fdinfo_client(proc, 200, 4, pdev=pdev, cid=2, ccs=50, total=1000, vram=10_000_000)

    times = iter([100.0, 102.0])
    p = GpuProvider(
        enabled=True,
        drm_root=str(drm),
        clock=lambda: next(times),
        proc_root=str(proc),
        pdev_resolver=lambda _card_path: pdev,  # bypass realpath on the fake tree
    )

    snap1 = p.sample()
    g1 = snap1.gpus[0]
    assert g1.vendor == "intel"
    # VRAM available on the first sample (sum across unique clients, KiB->bytes).
    assert g1.mem_used == (20_000_000 + 10_000_000) * 1024
    assert g1.mem_total is None  # total VRAM capacity unknown on xe
    assert g1.util_gpu is None  # first sample -> no util yet

    # Advance the compute counter: client 1 ccs 100->600 (+500), engine elapsed
    # cycles 1000->2000 (+1000) -> 50% util on the second sample.
    _fdinfo_client(proc, 100, 3, pdev=pdev, cid=1, ccs=600, total=2000, vram=20_000_000)
    _fdinfo_client(proc, 200, 4, pdev=pdev, cid=2, ccs=50, total=2000, vram=10_000_000)
    snap2 = p.sample()
    g2 = snap2.gpus[0]
    assert g2.util_gpu == 50.0
    assert g2.mem_used == (20_000_000 + 10_000_000) * 1024


def test_provider_intel_no_fdinfo_leaves_util_vram_none(tmp_path):  # noqa: ANN001
    # No GPU clients in the fake /proc -> util/VRAM stay None, sysfs fields still
    # populate, and nothing raises.
    drm = _intel_card(tmp_path / "sys")
    proc = tmp_path / "proc"
    proc.mkdir()
    p = GpuProvider(
        enabled=True,
        drm_root=str(drm),
        clock=lambda: 100.0,
        proc_root=str(proc),
        pdev_resolver=lambda _card_path: "0000:06:00.0",
    )
    g = p.sample().gpus[0]
    assert g.util_gpu is None
    assert g.mem_used is None and g.mem_total is None
    assert g.temp_c == 57.0  # sysfs temp still read
    assert g.fan_rpm == 1060


def _add_gtidle(drm, idle_ms, *, gt="tile0/gt0"):  # noqa: ANN001
    """Add a gtidle idle-residency counter under card0 in the fake DRM tree."""
    from pathlib import Path

    p = drm / "card0" / "device" / Path(gt) / "gtidle"
    p.mkdir(parents=True, exist_ok=True)
    (p / "idle_residency_ms").write_text(f"{idle_ms}\n")


def test_provider_intel_util_from_gtidle_no_root(tmp_path):  # noqa: ANN001
    # gtidle gives util without root (no fdinfo clients present at all).
    drm = _intel_card(tmp_path / "sys")
    _add_gtidle(drm, 5000)
    proc = tmp_path / "proc"
    proc.mkdir()
    times = iter([100.0, 101.0])
    p = GpuProvider(
        enabled=True,
        drm_root=str(drm),
        clock=lambda: next(times),
        proc_root=str(proc),
        pdev_resolver=lambda _card_path: "0000:06:00.0",
    )
    g1 = p.sample().gpus[0]
    assert g1.util_gpu is None  # first sample -> no idle delta yet
    assert g1.mem_used is None  # no fdinfo clients

    # idle advances 300ms over a 1000ms wall window -> 70% busy, no root needed.
    _add_gtidle(drm, 5300)
    g2 = p.sample().gpus[0]
    assert g2.util_gpu == 70.0
    assert g2.mem_used is None  # VRAM still root-gated / absent


def test_provider_intel_gtidle_is_primary_over_fdinfo(tmp_path):  # noqa: ANN001
    # When both gtidle and fdinfo report util, gtidle (non-root) wins; VRAM
    # still comes from fdinfo.
    drm = _intel_card(tmp_path / "sys")
    _add_gtidle(drm, 5000)
    proc = tmp_path / "proc"
    pdev = "0000:06:00.0"
    _fdinfo_client(proc, 100, 3, pdev=pdev, cid=1, ccs=100, total=1000, vram=20_000_000)
    times = iter([100.0, 101.0])
    p = GpuProvider(
        enabled=True,
        drm_root=str(drm),
        clock=lambda: next(times),
        proc_root=str(proc),
        pdev_resolver=lambda _card_path: pdev,
    )
    p.sample()  # prime both gtidle and fdinfo carries
    # gtidle: +300ms idle / 1000ms -> 70% busy. fdinfo: ccs +500 / total +1000 -> 50%.
    _add_gtidle(drm, 5300)
    _fdinfo_client(proc, 100, 3, pdev=pdev, cid=1, ccs=600, total=2000, vram=20_000_000)
    g = p.sample().gpus[0]
    assert g.util_gpu == 70.0  # gtidle (non-root) is primary
    assert g.mem_used == 20_000_000 * 1024  # VRAM from fdinfo


def test_provider_intel_fdinfo_util_fallback_when_gtidle_none(tmp_path):  # noqa: ANN001
    # No gtidle counters -> util falls back to fdinfo.
    drm = _intel_card(tmp_path / "sys")  # no gtidle tree added
    proc = tmp_path / "proc"
    pdev = "0000:06:00.0"
    _fdinfo_client(proc, 100, 3, pdev=pdev, cid=1, ccs=100, total=1000, vram=20_000_000)
    times = iter([100.0, 101.0])
    p = GpuProvider(
        enabled=True,
        drm_root=str(drm),
        clock=lambda: next(times),
        proc_root=str(proc),
        pdev_resolver=lambda _card_path: pdev,
    )
    p.sample()
    _fdinfo_client(proc, 100, 3, pdev=pdev, cid=1, ccs=600, total=2000, vram=20_000_000)
    g = p.sample().gpus[0]
    assert g.util_gpu == 50.0  # fdinfo fallback (gtidle unavailable)
    assert g.mem_used == 20_000_000 * 1024


def test_sample_degrades_when_backend_read_raises(tmp_path, monkeypatch):  # noqa: ANN001
    # A backend read that raises unexpectedly must degrade to an unavailable
    # snapshot, not propagate out of sample() and crash the app's tick loop.
    drm = _intel_card(tmp_path)
    p = GpuProvider(enabled=True, drm_root=str(drm), clock=lambda: 100.0)

    import vllmstat.providers.gpu as gpumod

    def boom(*_a, **_k):
        raise TypeError("can't concat NoneType to bytes")

    monkeypatch.setattr(gpumod, "read_intel_sysfs", boom)

    snap = p.sample()  # must not raise
    assert snap.available is False
    assert snap.error


def test_provider_no_cards_no_nvml_reports_unavailable(tmp_path):  # noqa: ANN001
    empty = tmp_path / "drm"
    empty.mkdir()
    # Force the NVML import to fail so we exercise the "nothing found" path.
    p = GpuProvider(enabled=True, drm_root=str(empty))
    p._nvml = None
    p._mode = "nvidia-smi"  # skip the NVML import attempt

    import vllmstat.providers.gpu as gpumod

    orig = gpumod.shutil.which
    gpumod.shutil.which = lambda _name: None  # type: ignore[assignment]
    try:
        snap = p.sample()
    finally:
        gpumod.shutil.which = orig  # type: ignore[assignment]
    assert snap.available is False
    assert snap.error
