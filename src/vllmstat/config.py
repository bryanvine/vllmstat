from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from vllmstat import __version__
from vllmstat.core.energy import EnergyConfig
from vllmstat.core.state import Instance


@dataclass
class Config:
    urls: list[str] = field(default_factory=list)
    metrics_path: str = "/metrics"
    interval: float = 1.0
    api_key: str | None = None
    gpu: bool = True
    mock: bool = False
    once: bool = False
    json: bool = False
    config_path: str | None = None
    # None = auto (discover unless explicit targets were given); True/False = forced by flag.
    discover_docker: bool | None = None
    discover_proc: bool | None = None  # same auto/forced semantics, for bare host processes
    combined: bool = False
    instances: list[Instance] = field(default_factory=list)
    logs: str | None = None
    proxy: str | None = None
    energy: EnergyConfig = field(default_factory=EnergyConfig)

    @property
    def url(self) -> str:
        return self.urls[0] if self.urls else "http://localhost:8000"

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(prog="vllmstat", description="nvtop for vLLM")
        p.add_argument("-u", "--url", action="append", dest="urls", default=None, metavar="URL")
        p.add_argument("--metrics-path", default="/metrics")
        p.add_argument("-i", "--interval", type=float, default=1.0)
        p.add_argument("--api-key", default=None)
        p.add_argument("--no-gpu", dest="gpu", action="store_false", default=True)
        p.add_argument("--mock", action="store_true", default=False)
        p.add_argument("--once", action="store_true", default=False)
        p.add_argument("--json", action="store_true", default=False)
        p.add_argument("--config", dest="config_path", default=None)
        p.add_argument(
            "--discover-docker",
            dest="discover_docker",
            action="store_true",
            default=None,
            help="force Docker discovery even when --url/--config are given (default: auto)",
        )
        p.add_argument(
            "--no-discover-docker",
            dest="discover_docker",
            action="store_false",
            help="never scan the local Docker daemon for vLLM containers",
        )
        p.add_argument(
            "--discover-proc",
            dest="discover_proc",
            action="store_true",
            default=None,
            help="force host-process discovery even when --url/--config are given (default: auto)",
        )
        p.add_argument(
            "--no-discover-proc",
            dest="discover_proc",
            action="store_false",
            help="never scan local processes for `vllm serve` servers",
        )
        p.add_argument(
            "--combined",
            dest="combined",
            action="store_true",
            default=False,
            help="open the combined (all instances) dashboard; with --once --json, emit it",
        )
        p.add_argument("--logs", dest="logs", default=None)
        p.add_argument("--proxy", dest="proxy", default=None)
        p.add_argument("--version", action="version", version=f"vllmstat {__version__}")
        return p

    @classmethod
    def from_sources(cls, argv: list[str], env: dict[str, str]) -> Config:
        ns = cls.build_parser().parse_args(argv)
        api_key = ns.api_key or env.get("VLLM_API_KEY")
        return cls(
            urls=list(ns.urls or []),
            metrics_path=ns.metrics_path,
            interval=ns.interval,
            api_key=api_key,
            gpu=ns.gpu,
            mock=ns.mock,
            once=ns.once,
            json=ns.json,
            config_path=ns.config_path,
            discover_docker=ns.discover_docker,
            discover_proc=ns.discover_proc,
            combined=ns.combined,
            logs=ns.logs,
            proxy=ns.proxy,
        )
