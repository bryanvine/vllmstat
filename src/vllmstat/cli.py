from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

from vllmstat.config import Config
from vllmstat.core.metrics import MetricsEngine
from vllmstat.core.parse import parse_metrics
from vllmstat.providers.discover_docker import discover_docker
from vllmstat.providers.mock import MockProvider
from vllmstat.snapshot_json import snapshot_to_dict


def port_responding(url: str, timeout: float = 0.4) -> bool:
    import socket
    from urllib.parse import urlparse

    p = urlparse(url if "://" in url else "http://" + url)
    host = p.hostname or "localhost"
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_instances(cfg: Config, env: dict[str, str]) -> Config:
    from vllmstat.core.config_file import find_config, load_config
    from vllmstat.core.resolve import instance_from_dict, local_hostnames, resolve_fleet

    local_names = local_hostnames()
    config_instances = []
    config_globals: dict = {}
    path = find_config(cfg.config_path, env)
    if path:
        try:
            raw, config_globals = load_config(path)
            config_instances = [
                instance_from_dict(
                    r,
                    defaults_api_key=cfg.api_key,
                    defaults_metrics_path=cfg.metrics_path,
                    local_names=local_names,
                )
                for r in raw
            ]
        except (OSError, ValueError) as e:
            print(f"vllmstat: ignoring config {path}: {e}", file=sys.stderr)
    # Config-file global keys fill in for CLI flags left at their default
    # (an explicitly-passed non-default flag still wins).
    interval = config_globals.get("interval")
    if isinstance(interval, bool):  # TOML booleans are not valid intervals
        interval = None
    if cfg.interval == 1.0 and isinstance(interval, (int, float)):
        cfg.interval = float(interval)
    gpu = config_globals.get("gpu")
    if cfg.gpu is True and isinstance(gpu, bool):
        cfg.gpu = gpu
    energy_tbl = config_globals.get("energy")
    if isinstance(energy_tbl, dict):
        from vllmstat.core.energy import parse_energy_config
        try:
            cfg.energy = parse_energy_config(energy_tbl)
        except ValueError as e:
            print(f"vllmstat: ignoring [energy] config: {e}", file=sys.stderr)
    docker_instances = discover_docker() if cfg.discover_docker else []
    default_url = "http://localhost:8000"
    if (
        not config_instances
        and not docker_instances
        and not cfg.urls
        and not cfg.mock
        and not cfg.discover_docker  # explicit discovery already ran above; don't re-probe
    ):
        if not port_responding(default_url):
            found = discover_docker()
            if found:
                docker_instances = found
                print(
                    f"vllmstat: {default_url} not responding — "
                    f"found {len(found)} vLLM container(s) via Docker",
                    file=sys.stderr,
                )
    cfg.instances = resolve_fleet(
        config_instances,
        docker_instances,
        cfg.urls,
        defaults_api_key=cfg.api_key,
        defaults_metrics_path=cfg.metrics_path,
        local_names=local_names,
    )
    if cfg.logs:
        cfg.instances = [replace(i, logs=i.logs or cfg.logs) for i in cfg.instances]
    return cfg


def _run_once_fleet(cfg: Config) -> int:
    import asyncio

    from vllmstat.core.fleet import Fleet, InstanceRuntime
    from vllmstat.core.state import GpuSnapshot

    async def go():
        if cfg.mock:
            from vllmstat.providers.mock import MockProvider, MockVllmProvider

            rts = [
                InstanceRuntime(i, provider=MockVllmProvider(MockProvider())) for i in cfg.instances
            ]
        else:
            rts = [InstanceRuntime(i) for i in cfg.instances]
        fleet = Fleet([], runtimes=rts)
        await fleet.poll(GpuSnapshot(), 0.0)
        time.sleep(min(cfg.interval, 1.0))
        fs = await fleet.poll(GpuSnapshot(), 1.0)
        await fleet.aclose()
        return fs

    fs = asyncio.run(go())
    out = [
        {
            "name": inst.name,
            "url": inst.url,
            "locality": inst.locality,
            "snapshot": snapshot_to_dict(snap),
        }
        for inst, snap in fs.items
    ]
    print(json.dumps(out, default=str))
    return 0


def run_once_json(cfg: Config) -> int:
    if len(cfg.instances) > 1:
        return _run_once_fleet(cfg)
    if cfg.mock:
        eng = MetricsEngine(dims=None, max_model_len=None)
        mp = MockProvider()
        eng.derive(parse_metrics(mp.metrics_text()), now=0.0)
        snap = eng.derive(parse_metrics(mp.metrics_text()), now=1.0)
    else:
        import asyncio

        from vllmstat.model_dims import load_model_dims
        from vllmstat.providers.vllm import VllmProvider

        async def _go():
            inst = cfg.instances[0] if cfg.instances else None
            url = inst.url if inst else cfg.url
            metrics_path = inst.metrics_path if inst else cfg.metrics_path
            api_key = inst.api_key if inst else cfg.api_key
            p = VllmProvider(base_url=url, metrics_path=metrics_path, api_key=api_key)
            info = await p.fetch_model_info()
            r0 = await p.fetch_metrics()
            time.sleep(min(cfg.interval, 1.0))
            r1 = await p.fetch_metrics()
            await p.aclose()
            return info, r0, r1

        info, r0, r1 = asyncio.run(_go())
        if not r1.fetched_ok:
            print(json.dumps({"error": r1.error}), file=sys.stderr)
            return 1
        md = load_model_dims(info.root, info.max_model_len)
        eng = MetricsEngine(dims=md.dims, max_model_len=md.max_model_len)
        eng.derive(parse_metrics(r0.text), now=0.0)
        snap = eng.derive(parse_metrics(r1.text), now=1.0)
    print(json.dumps(snapshot_to_dict(snap), default=str))
    return 0


def _daemon_main(argv: list[str], env: dict[str, str]) -> int:
    import argparse
    from pathlib import Path

    from vllmstat.core.service import install_unit, resolve_store_path, uninstall_unit, unit_path
    from vllmstat.core.store import Store

    p = argparse.ArgumentParser(prog="vllmstat daemon")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "install", "uninstall", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", dest="config_path", default=None)
        sp.add_argument("--store", dest="store", default=None)
        sp.add_argument("--system", dest="system", action="store_true", default=False)
        sp.add_argument("--user", dest="user", action="store_true", default=False)
        sp.add_argument("--json", dest="json", action="store_true", default=False)
    ns = p.parse_args(argv)

    cfg = Config.from_sources((["--config", ns.config_path] if ns.config_path else []), env)
    resolve_instances(cfg, env)
    if ns.store:
        from vllmstat.core.energy import replace_store
        cfg.energy = replace_store(cfg.energy, ns.store)

    if ns.cmd == "run":
        from vllmstat.daemon import run
        return run(cfg)

    # install/uninstall default to system unless --user is given (--system is explicit)
    system = ns.system or not ns.user

    if ns.cmd == "install":
        try:
            path = install_unit(system=system)
        except PermissionError:
            print("vllmstat: need root to install a system unit (try --user or sudo)",
                  file=sys.stderr)
            return 1
        scope = "" if system else "--user "
        print(f"wrote {path}\nenable with:\n  systemctl {scope}daemon-reload\n"
              f"  systemctl {scope}enable --now vllmstat")
        return 0
    if ns.cmd == "uninstall":
        removed = uninstall_unit(system=system)
        print(f"removed {unit_path(system=system)}" if removed else "no unit installed")
        return 0

    # status
    path = resolve_store_path(cfg, for_write=False)
    if not Path(path).expanduser().exists():
        print("no energy store yet (start the daemon: vllmstat daemon run)")
        return 0
    store = Store.open(path, read_only=True)
    try:
        view = store.read_view(now=time.time(), currency=cfg.energy.currency)
        gpus = store.totals_gpu()
    finally:
        store.close()
    if ns.json:
        print(json.dumps({
            "today_kwh": view.today_kwh, "today_cost": view.today_cost,
            "alltime_kwh": view.alltime_kwh, "alltime_cost": view.alltime_cost,
            "gpus": gpus,
        }, default=str))
    else:
        cur = cfg.energy.currency
        tc = f" ({cur}{view.today_cost:.2f})" if view.today_cost is not None else ""
        ac = f" ({cur}{view.alltime_cost:.2f})" if view.alltime_cost is not None else ""
        print(f"today:    {view.today_kwh:.2f} kWh{tc}")
        print(f"all-time: {view.alltime_kwh:.2f} kWh{ac}")
        for g in gpus:
            print(f"  GPU{g['gpu_idx']}: {g['kwh']:.2f} kWh")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    env = dict(os.environ)
    if argv and argv[0] == "daemon":
        return _daemon_main(argv[1:], env)
    cfg = Config.from_sources(argv, env)
    if cfg.proxy:
        from vllmstat.providers.proxy import parse_proxy_addr

        try:
            parse_proxy_addr(cfg.proxy)
        except ValueError as e:
            print(f"vllmstat: {e}", file=sys.stderr)
            return 2
    resolve_instances(cfg, env)
    if cfg.once and cfg.json:
        return run_once_json(cfg)
    from vllmstat.app import run_app  # imported lazily so --once/--json needs no Textual

    return run_app(cfg)
