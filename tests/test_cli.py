from vllmstat.cli import main


def test_daemon_install_writes_unit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    rc = main(["daemon", "install", "--user"])
    assert rc == 0
    unit = tmp_path / "systemd" / "user" / "vllmstat.service"
    assert unit.exists() and "daemon run" in unit.read_text()
    out = capsys.readouterr().out
    assert "systemctl" in out


def test_daemon_uninstall(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    main(["daemon", "install", "--user"])
    rc = main(["daemon", "uninstall", "--user"])
    assert rc == 0
    assert not (tmp_path / "systemd" / "user" / "vllmstat.service").exists()


def test_daemon_status_no_store(tmp_path, monkeypatch, capsys):
    # point both XDG dirs at an empty tmp so no store is found
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    rc = main(["daemon", "status", "--store", str(tmp_path / "missing.db")])
    out = capsys.readouterr().out
    assert rc == 0 and "no energy store" in out.lower()


def test_daemon_status_reads_store(tmp_path, monkeypatch, capsys):
    from vllmstat.core.energy import GpuEnergy, InstanceEnergy
    from vllmstat.core.store import Store

    db = tmp_path / "e.db"
    s = Store.open(str(db))
    s.record(1782648000.0, [GpuEnergy(0, 200.0, 1.5, 0.30)], [InstanceEnergy("a", 1.5, 0.30)])
    s.close()
    rc = main(["daemon", "status", "--store", str(db), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    import json as _j
    data = _j.loads(out)
    assert data["alltime_kwh"] == 1.5 and data["gpus"][0]["gpu_idx"] == 0


def test_daemon_status_human_output(tmp_path, capsys):
    from vllmstat.cli import main
    from vllmstat.core.energy import GpuEnergy, InstanceEnergy
    from vllmstat.core.store import Store

    db = tmp_path / "e.db"
    s = Store.open(str(db))
    s.record(1782648000.0, [GpuEnergy(0, 200.0, 1.5, 0.30)], [InstanceEnergy("a", 1.5, 0.30)])
    s.close()
    rc = main(["daemon", "status", "--store", str(db)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "all-time: 1.50 kWh ($0.30)" in out
    assert "GPU0: 1.50 kWh" in out
