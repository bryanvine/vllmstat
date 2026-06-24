from __future__ import annotations

import os
from pathlib import Path

from vllmstat.config import Config

SYSTEM_STORE = "/var/lib/vllmstat/vllmstat.db"


def _user_store() -> str:
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return str(Path(base) / "vllmstat" / "vllmstat.db")


USER_STORE = _user_store()


def resolve_store_path(cfg: Config, *, for_write: bool) -> str:
    """Resolve the energy DB path.

    `--store`/config override wins. Otherwise prefer the system store when its
    directory is writable (daemon as root); else fall back to the per-user store.
    For reads, prefer the system store if it exists, else the user store — so a
    user-run TUI still finds a user-run daemon's DB.
    """
    if cfg.energy.store:
        return cfg.energy.store
    sys_dir = Path(SYSTEM_STORE).parent
    if for_write:
        if os.access(sys_dir, os.W_OK) or (not sys_dir.exists() and os.access("/var/lib", os.W_OK)):
            return SYSTEM_STORE
        return USER_STORE
    return SYSTEM_STORE if Path(SYSTEM_STORE).exists() else USER_STORE


def unit_path(*, system: bool) -> str:
    if system:
        return "/etc/systemd/system/vllmstat.service"
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return str(Path(base) / "systemd" / "user" / "vllmstat.service")


def systemd_unit(*, exec_path: str, system: bool) -> str:
    target = "multi-user.target" if system else "default.target"
    return (
        "[Unit]\n"
        "Description=vllmstat energy/stats collector\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={exec_path} daemon run\n"
        "Restart=on-failure\n"
        "RestartSec=5\n\n"
        "[Install]\n"
        f"WantedBy={target}\n"
    )


def install_unit(*, system: bool, exec_path: str | None = None) -> str:
    """Write the unit file and return its path. Raises PermissionError without rights."""
    import shutil

    exec_path = exec_path or shutil.which("vllmstat") or "vllmstat"
    path = unit_path(system=system)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(systemd_unit(exec_path=exec_path, system=system))
    return path


def uninstall_unit(*, system: bool) -> bool:
    path = Path(unit_path(system=system))
    if path.exists():
        path.unlink()
        return True
    return False
