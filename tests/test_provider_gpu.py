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
