from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable

from vllmstat.core.resolve import normalize_url
from vllmstat.core.state import Instance

VLLM_MARKERS = ("vllm", "api_server", "openai.api_server")


def is_vllm_container(row: dict) -> bool:
    blob = f"{row.get('Image', '')} {row.get('Command', '')}".lower()
    return any(m in blob for m in VLLM_MARKERS)


def parse_ps(out: str) -> list[dict]:
    rows: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def host_port_from_inspect(insp: dict, *, container_port: str = "8000") -> int | None:
    ports = (insp.get("NetworkSettings") or {}).get("Ports") or {}
    binds = ports.get(f"{container_port}/tcp")
    if binds:
        return int(binds[0]["HostPort"])
    for key, b in ports.items():
        if key.endswith("/tcp") and b:
            return int(b[0]["HostPort"])
    return None


def gpus_from_inspect(insp: dict, *, host_gpu_count: int | None = None) -> tuple[int, ...]:
    hc = insp.get("HostConfig") or {}
    for req in hc.get("DeviceRequests") or []:
        caps = req.get("Capabilities") or []
        if any("gpu" in c for grp in caps for c in grp):
            ids = req.get("DeviceIDs") or []
            got = tuple(int(i) for i in ids if str(i).isdigit())
            if got:
                return got
            if req.get("Count") in (-1, None) and host_gpu_count:
                return tuple(range(host_gpu_count))
    for e in (insp.get("Config") or {}).get("Env") or []:
        if e.startswith("NVIDIA_VISIBLE_DEVICES="):
            val = e.split("=", 1)[1]
            if val in ("all", "") and host_gpu_count:
                return tuple(range(host_gpu_count))
            got = tuple(int(t) for t in val.split(",") if t.strip().isdigit())
            if got:
                return got
    return ()


def _name(insp: dict, row: dict) -> str:
    n = (insp.get("Name") or "").lstrip("/")
    return n or row.get("Names") or (row.get("ID") or "")[:12]


def build_instances(
    ps_rows: list[dict],
    inspect_by_id: dict[str, dict],
    *,
    host_gpu_count: int | None = None,
    default_metrics_path: str = "/metrics",
) -> list[Instance]:
    out: list[Instance] = []
    for row in ps_rows:
        if not is_vllm_container(row):
            continue
        cid = row.get("ID") or row.get("Id") or ""
        insp = inspect_by_id.get(cid) or {}
        port = host_port_from_inspect(insp)
        if not port:
            continue
        out.append(
            Instance(
                name=_name(insp, row),
                url=normalize_url(f"http://localhost:{port}"),
                metrics_path=default_metrics_path,
                gpus=gpus_from_inspect(insp, host_gpu_count=host_gpu_count),
                locality="local",
            )
        )
    return out


def discover_docker(
    *,
    run: Callable[[list[str]], str] | None = None,
    host_gpu_count: int | None = None,
) -> list[Instance]:
    _run = run or _default_run
    try:
        rows = parse_ps(_run(["docker", "ps", "--no-trunc", "--format", "{{json .}}"]))
        inspect_by_id: dict[str, dict] = {}
        for r in rows:
            if not is_vllm_container(r):
                continue
            cid = r.get("ID") or r.get("Id") or ""
            if not cid:
                continue
            data = json.loads(_run(["docker", "inspect", cid]))
            if isinstance(data, list) and data:
                inspect_by_id[cid] = data[0]
        return build_instances(rows, inspect_by_id, host_gpu_count=host_gpu_count)
    except Exception:  # noqa: BLE001 - discovery is best-effort, never crash the app
        return []


def _default_run(cmd: list[str]) -> str:
    if not shutil.which("docker"):
        raise FileNotFoundError("docker not found")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout
