"""Discover vLLM servers running as plain host processes (systemd units, tmux, venvs).

Docker discovery only sees containers; a bare ``vllm serve`` launched from a venv
is invisible to it. Here we walk ``/proc``, pick out vLLM API-server command
lines, and read ``--host`` / ``--port`` straight from argv. ``/proc/<pid>/cmdline``
and ``/proc/<pid>/cgroup`` are world-readable, so this works even when vLLM runs
as another user (socket→pid mapping would need root, argv doesn't).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from vllmstat.core.resolve import normalize_url
from vllmstat.core.state import Instance

# vLLM's own defaults when --host / --port are omitted.
DEFAULT_PORT = 8000
_WILDCARD_HOSTS = {"", "0.0.0.0", "::", "[::]"}
# A containerised vLLM also shows up in the host's /proc, but its argv port is the
# in-container one; Docker discovery owns those.
_CONTAINER_CGROUP_MARKERS = ("docker", "containerd", "libpod", "kubepods", "lxc", "crio")


def _flag(argv: list[str], name: str) -> str | None:
    """Value of ``--name X`` or ``--name=X`` (last occurrence wins, like argparse)."""
    val: str | None = None
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            val = argv[i + 1]
        elif a.startswith(name + "="):
            val = a.split("=", 1)[1]
    return val


def _flag_values(argv: list[str], name: str) -> list[str]:
    """All values following ``--name`` up to the next ``-``-prefixed token (nargs='+')."""
    for i, a in enumerate(argv):
        if a.startswith(name + "="):
            return [a.split("=", 1)[1]]
        if a == name:
            out: list[str] = []
            for b in argv[i + 1 :]:
                if b.startswith("-"):
                    break
                out.append(b)
            return out
    return []


def serve_args(argv: list[str]) -> list[str] | None:
    """The args after the vLLM entrypoint, or None if argv isn't a vLLM API server."""
    for i, a in enumerate(argv):
        base = os.path.basename(a)
        # `vllm serve ...` (console script, possibly run as `python .../bin/vllm serve`)
        if base == "vllm" and i + 1 < len(argv) and argv[i + 1] == "serve":
            return argv[i + 2 :]
        # `python -m vllm.entrypoints.openai.api_server ...`
        if a in ("vllm.entrypoints.openai.api_server", "vllm.entrypoints.api_server"):
            return argv[i + 1 :]
    return None


def instance_from_argv(
    argv: list[str], *, default_metrics_path: str = "/metrics"
) -> Instance | None:
    args = serve_args(argv)
    if args is None or "--uds" in args or any(a.startswith("--uds=") for a in args):
        return None
    port_s = _flag(args, "--port")
    try:
        port = int(port_s) if port_s else DEFAULT_PORT
    except ValueError:
        return None
    host = _flag(args, "--host") or ""
    if host in _WILDCARD_HOSTS:
        host = "localhost"
    elif ":" in host and not host.startswith("["):
        host = f"[{host}]"
    names = _flag_values(args, "--served-model-name")
    model = args[0] if args and not args[0].startswith("-") else (_flag(args, "--model") or "")
    name = names[0] if names else (os.path.basename(model.rstrip("/")) or f"{host}:{port}")
    return Instance(
        name=name,
        url=normalize_url(f"http://{host}:{port}"),
        metrics_path=default_metrics_path,
        locality="local",
    )


def _in_container(cgroup_text: str) -> bool:
    return any(m in cgroup_text for m in _CONTAINER_CGROUP_MARKERS)


def discover_procs(
    *, proc_root: str = "/proc", default_metrics_path: str = "/metrics"
) -> list[Instance]:
    root = Path(proc_root)
    if not root.is_dir():
        return []  # not Linux / no procfs: nothing to discover
    found: dict[str, Instance] = {}
    try:
        pids = sorted((p for p in root.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    except OSError as e:
        print(f"vllmstat: process discovery failed: {e}", file=sys.stderr)
        return []
    for pdir in pids:
        try:
            raw = (pdir / "cmdline").read_bytes()
        except OSError:
            continue  # process exited or is hidden from us
        if b"vllm" not in raw:
            continue
        argv = [a.decode(errors="replace") for a in raw.split(b"\0") if a]
        inst = instance_from_argv(argv, default_metrics_path=default_metrics_path)
        if inst is None:
            continue
        try:
            if _in_container((pdir / "cgroup").read_text()):
                continue
        except OSError:
            pass
        # Wrappers (`uv run vllm serve`, a shell `exec`) can match twice; first pid wins.
        found.setdefault(inst.url, inst)
    return list(found.values())
