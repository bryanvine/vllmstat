from __future__ import annotations

from vllmstat.core.history import History
from vllmstat.core.state import Snapshot
from vllmstat.format import fmt_bytes, fmt_dur, fmt_pct, fmt_si, sparkline
from vllmstat.plot import braille_plot

# Default plot width when the panel width is unknown (e.g. first render).
_DEFAULT_PLOT_WIDTH = 30
_MIN_PLOT_WIDTH = 8


def _plot_width(width: int | None) -> int:
    if not width or width <= 0:
        return _DEFAULT_PLOT_WIDTH
    return max(_MIN_PLOT_WIDTH, int(width))


def header(s: Snapshot, *, url: str, interval: float, uptime: str) -> str:
    state = "● connected" if s.connected else "● down"
    models = ",".join(s.model_names) or "—"
    parts = f"vllmstat  {models} @ {url}  engines {s.engine_count}"
    return f"{parts}  {state}  up {uptime}  {interval:.1f}s"


def concurrency(s: Snapshot, h: History, *, width: int | None = None) -> str:
    pw = _plot_width(width)
    vals = list(h.series("running").values)
    plot = "\n".join(braille_plot(vals, width=pw, height=4, lo=0))
    seqs = f"  max-seqs {s.max_num_seqs}" if s.max_num_seqs else ""
    return (
        f"CONCURRENCY\n"
        f" running {s.running:.0f} · waiting {s.waiting:.0f} · "
        f"preempt {s.preempt_rate:.1f}/s{seqs}\n"
        f"{plot}\n"
        f" running · last {len(vals)}s"
    )


def throughput(s: Snapshot, h: History, *, width: int | None = None) -> str:
    pw = _plot_width(width)
    vals = list(h.series("gen_tps").values)
    plot = "\n".join(braille_plot(vals, width=pw, height=4, lo=0))
    tpi = f"{s.tokens_per_iter:.0f}" if s.tokens_per_iter else "—"
    return (
        f"THROUGHPUT\n"
        f" gen {s.gen_tps:.0f} tok/s · prompt {s.prompt_tps:.0f} tok/s · "
        f"tok/iter {tpi} · {s.req_rate:.1f} req/s\n"
        f"{plot}\n"
        f" gen tok/s · last {len(vals)}s"
    )


def cache_kv(s: Snapshot, h: History) -> str:
    hit_spark = sparkline(list(h.series("prefix_hit").values))
    src = (
        f"compute {fmt_pct(s.src_compute)} · "
        f"cache-hit {fmt_pct(s.src_cache_hit)} · ext {fmt_pct(s.src_external)}"
    )
    ratio = ""
    if s.kv_ratio and s.kv_ratio_kind != "none":
        tag = "~" if s.kv_ratio_kind == "nominal" else ""
        ratio = f"  {tag}{s.kv_ratio:.1f}x vs fp16"
    cap = fmt_si(s.kv_capacity_tokens) if s.kv_capacity_tokens else "—"
    used = fmt_si(s.kv_used_tokens) if s.kv_used_tokens is not None else "—"
    ctx = f" (fp16 full ctx {s.kv_fp16_full_ctx_gb:.1f}GB)" if s.kv_fp16_full_ctx_gb else ""
    return (
        f"CACHE & KV MEMORY\n"
        f" reuse  prefix hit {fmt_pct(s.prefix_hit_window)} ▕{hit_spark}▏ "
        f"life {fmt_pct(s.prefix_hit_lifetime)}   sources {src}\n"
        f" memory KV usage {fmt_pct(s.kv_usage)} ({used}/{cap} tok)   "
        f"{s.kv_dtype or '—'}{ratio}{ctx}"
    )


def _q(label: str, q) -> str:
    return f" {label:<6} {fmt_dur(q.p50):>7} {fmt_dur(q.p90):>7} {fmt_dur(q.p99):>7}"


def latency(s: Snapshot) -> str:
    head = f" {'':6} {'p50':>7} {'p90':>7} {'p99':>7}"
    return (
        "LATENCY (recent)\n"
        + head
        + "\n"
        + _q("TTFT", s.ttft)
        + "\n"
        + _q("TPOT", s.tpot)
        + "\n"
        + _q("e2e", s.e2e)
        + "\n"
        + _q("queue", s.queue)
    )


def specdecode(s: Snapshot) -> str:
    if not s.spec_active:
        return ""
    apd = f"{s.spec_accepted_per_draft:.2f}" if s.spec_accepted_per_draft is not None else "—"
    return f"SPEC DECODE  acceptance {fmt_pct(s.spec_acceptance)}  accepted/draft {apd}"


def efficiency(s: Snapshot) -> str:
    if not s.eff_active:
        return ""
    parts = []
    if s.gflops is not None:
        parts.append(f"{s.gflops:.0f} GFLOP/s")
    if s.gbps is not None:
        parts.append(f"{s.gbps:.0f} GB/s")
    if s.mfu is not None:
        parts.append(f"MFU {fmt_pct(s.mfu)}")
    return "EFFICIENCY  " + " · ".join(parts) if parts else ""


def gpu(s: Snapshot) -> str:
    if not s.gpu.available:
        return f"GPU  unavailable ({s.gpu.error or 'no NVML/nvidia-smi'})"
    lines = []
    for g in s.gpu.gpus:
        mem_pct = (g.mem_used / g.mem_total) if (g.mem_used is not None and g.mem_total) else None
        util = f"{g.util_gpu:.0f}%" if g.util_gpu is not None else "—"
        temp = f"{g.temp_c:.0f}°C" if g.temp_c is not None else "—"
        pwr_w = f"{g.power_w:.0f}" if g.power_w is not None else "—"
        pwr_lim = f"{g.power_limit_w:.0f}" if g.power_limit_w is not None else "—"
        label = f"{g.vendor} {g.name}".strip() if g.vendor else g.name
        parts = [
            f"GPU {g.index}  {label}  {util}  "
            f"{fmt_bytes(g.mem_used)}/{fmt_bytes(g.mem_total)} ({fmt_pct(mem_pct)})  "
            f"{temp}  {pwr_w}/{pwr_lim} W"
        ]
        if g.fan_rpm is not None:
            parts.append(f"  fan {g.fan_rpm} RPM")
        elif g.fan_pct is not None:
            parts.append(f"  fan {g.fan_pct:.0f}%")
        if g.clock_sm_mhz is not None or g.clock_mem_mhz is not None:
            sm = g.clock_sm_mhz if g.clock_sm_mhz is not None else "—"
            mem = g.clock_mem_mhz if g.clock_mem_mhz is not None else "—"
            parts.append(f"  clk {sm}/{mem} MHz")
        if g.util_gpu is None and g.mem_used is None:
            parts.append("  (util/VRAM need root — see README)")
        lines.append("".join(parts))
    return "\n".join(lines)
