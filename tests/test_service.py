from vllmstat.config import Config
from vllmstat.core.energy import EnergyConfig
from vllmstat.core.service import (
    SYSTEM_STORE,
    USER_STORE,
    resolve_store_path,
    systemd_unit,
    uninstall_unit,
    unit_path,
)


def test_systemd_unit_contains_exec_and_install():
    unit = systemd_unit(exec_path="/usr/local/bin/vllmstat", system=True)
    assert "ExecStart=/usr/local/bin/vllmstat daemon run" in unit
    assert "[Service]" in unit and "WantedBy=multi-user.target" in unit
    assert "Restart=on-failure" in unit


def test_systemd_unit_user_target():
    unit = systemd_unit(exec_path="vllmstat", system=False)
    assert "WantedBy=default.target" in unit


def test_unit_path_system_vs_user():
    assert unit_path(system=True) == "/etc/systemd/system/vllmstat.service"
    assert unit_path(system=False).endswith("/.config/systemd/user/vllmstat.service")


def test_unit_path_user_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert unit_path(system=False) == str(tmp_path / "systemd" / "user" / "vllmstat.service")


def test_resolve_store_path_explicit_override_wins(tmp_path):
    cfg = Config.from_sources([], {})
    cfg.energy = EnergyConfig(store=str(tmp_path / "x.db"))
    assert resolve_store_path(cfg, for_write=True) == str(tmp_path / "x.db")
    assert resolve_store_path(cfg, for_write=False) == str(tmp_path / "x.db")


def test_resolve_store_path_default_is_known_location():
    cfg = Config.from_sources([], {})
    # no override -> one of the two known defaults (which one depends on perms/existence)
    assert resolve_store_path(cfg, for_write=True) in (SYSTEM_STORE, USER_STORE)
    assert resolve_store_path(cfg, for_write=False) in (SYSTEM_STORE, USER_STORE)


def test_resolve_store_path_user_respects_xdg_state(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    cfg = Config.from_sources([], {})
    result = resolve_store_path(cfg, for_write=False)
    # when the system store doesn't exist, the user path honors the patched XDG_STATE_HOME
    from pathlib import Path
    if not Path("/var/lib/vllmstat/vllmstat.db").exists():
        assert result == str(tmp_path / "vllmstat" / "vllmstat.db")


def test_install_and_uninstall_user_unit(monkeypatch, tmp_path):
    from vllmstat.core.service import install_unit

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = install_unit(system=False, exec_path="vllmstat")
    from pathlib import Path
    assert Path(path).exists() and "daemon run" in Path(path).read_text()
    assert uninstall_unit(system=False) is True
    assert not Path(path).exists()
    assert uninstall_unit(system=False) is False  # already gone
