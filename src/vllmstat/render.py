from __future__ import annotations

import time

from vllmstat.core.history import History
from vllmstat.core.state import FleetSnapshot, Instance, Snapshot
from vllmstat.core.tee import TeeEvent
from vllmstat.format import fmt_bytes, fmt_dur, fmt_dur_hms, fmt_pct, fmt_si, sparkline
from vllmstat.plot import braille_plot

# Default plot width when the panel width is unknown (e.g. first render).
_DEFAULT_PLOT_WIDTH = 30
_MIN_PLOT_WIDTH = 8


def _plot_width(width: int | None) -> int:
    if not width or width <= 0:
        return _DEFAULT_PLOT_WIDTH
    # Leave a small right margin so braille cells never touch the panel border.
    return max(_MIN_PLOT_WIDTH, int(width) - 2)


def header(s: Snapshot, *, url: str, interval: float, uptime: str) -> str:
    state = "● connected" if s.connected else "● down"
    models = ",".join(s.model_names) or "—"
    parts = f"vllmstat  {models} @ {url}  engines {s.engine_count}"
    return f"{parts}  {state}  up {uptime}  {interval:.1f}s"


def _series_plot(h: History, name: str, *, width: int, caption: str) -> str:
    """A 4-row braille plot of one history series with a labelled caption."""
    vals = list(h.series(name).values)
    plot = "\n".join(braille_plot(vals, width=width, height=4, lo=0))
    return f"{plot}\n {caption} · last {len(vals)}s"


def concurrency(s: Snapshot, h: History, *, width: int | None = None) -> str:
    pw = _plot_width(width)
    seqs = f"  max-seqs {s.max_num_seqs}" if s.max_num_seqs else ""
    return (
        f"CONCURRENCY\n"
        f" running {s.running:.0f} · waiting {s.waiting:.0f} · "
        f"preempt {s.preempt_rate:.1f}/s{seqs}\n"
        f"{_series_plot(h, 'running', width=pw, caption='running')}\n"
        f"{_series_plot(h, 'waiting', width=pw, caption='waiting')}"
    )


def throughput(s: Snapshot, h: History, *, width: int | None = None) -> str:
    pw = _plot_width(width)
    tpi = f"{s.tokens_per_iter:.0f}" if s.tokens_per_iter else "—"
    return (
        f"THROUGHPUT\n"
        f" gen {s.gen_tps:.0f} tok/s · prompt {s.prompt_tps:.0f} tok/s · "
        f"tok/iter {tpi} · {s.req_rate:.1f} req/s\n"
        f"{_series_plot(h, 'gen_tps', width=pw, caption='gen tok/s')}\n"
        f"{_series_plot(h, 'prompt_tps', width=pw, caption='prompt tok/s')}"
    )


def cache_kv(s: Snapshot, h: History) -> str:
    # Cap the inline sparkline so the line never grows long enough to wrap.
    hit_spark = sparkline(list(h.series("prefix_hit").values)[-16:])
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


def session(s: Snapshot) -> str:
    return (
        f"SESSION (while serving)\n"
        f" decode avg {fmt_si(s.avg_decode_tps)} tok/s · "
        f"prefill/pp avg {fmt_si(s.avg_prefill_tps)} tok/s · "
        f"active {fmt_pct(s.session_active_frac)} "
        f"({fmt_dur_hms(s.session_active_s)} busy / {fmt_dur_hms(s.session_idle_s)} idle)\n"
        f" {s.session_requests} reqs · {fmt_si(s.avg_gen_tokens_per_req)} gen tok/req · "
        f"totals {fmt_si(s.session_gen_tokens)} gen · {fmt_si(s.session_prompt_tokens)} prompt"
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
        + "\n"
        + _q("prefill", s.prefill)
        + "\n"
        + _q("decode", s.decode)
    )


def request_shape(s: Snapshot) -> str:
    if s.prompt_len.mean is None and s.gen_len.mean is None:
        return ""

    def row(label: str, q) -> str:
        avg = fmt_si(q.mean) if q.mean is not None else "—"
        p50 = fmt_si(q.p50) if q.p50 is not None else "—"
        p90 = fmt_si(q.p90) if q.p90 is not None else "—"
        return f" {label:<7} avg {avg:>5}  p50 {p50:>5}  p90 {p90:>5} tok"

    return "REQUEST SHAPE (recent)\n" + row("prompt", s.prompt_len) + "\n" + row("gen", s.gen_len)


def outcomes(s: Snapshot) -> str:
    if not s.finish_reasons and s.goodput_ttft is None and s.goodput_tpot is None:
        return ""
    parts = []
    if s.finish_reasons:
        ordered = sorted(s.finish_reasons.items(), key=lambda kv: kv[1], reverse=True)
        parts.append(" " + " · ".join(f"{k} {fmt_pct(v)}" for k, v in ordered))
    gp = []
    if s.goodput_ttft is not None:
        gp.append(f"TTFT<{fmt_dur(s.ttft_slo_s)} {fmt_pct(s.goodput_ttft)}")
    if s.goodput_tpot is not None:
        gp.append(f"TPOT<{fmt_dur(s.tpot_slo_s)} {fmt_pct(s.goodput_tpot)}")
    if gp:
        parts.append(" goodput  " + " · ".join(gp))
    return "OUTCOMES (recent)\n" + "\n".join(parts)


def specdecode(s: Snapshot) -> str:
    if not s.spec_active:
        return ""
    apd = f"{s.spec_accepted_per_draft:.2f}" if s.spec_accepted_per_draft is not None else "—"
    return f"SPEC DECODE  acceptance {fmt_pct(s.spec_acceptance)}  accepted/draft {apd}"


def _gpu_cell(s: Snapshot, inst: Instance) -> str:
    if inst.locality == "remote":
        return "(remote)"
    if not s.gpu.available or not s.gpu.gpus:
        return "—"
    idxs = ",".join(str(g.index) for g in s.gpu.gpus)
    g0 = s.gpu.gpus[0]
    util = f"{g0.util_gpu:.0f}%" if g0.util_gpu is not None else "—"
    return f"G{idxs} {g0.vendor} {util}".strip()


def fleet_overview(
    fleet: FleetSnapshot,
    selected: int,
    *,
    width: int | None = None,
    uptime: str = "",
    interval: float = 1.0,
    show_gpu: bool = True,
) -> str:
    n = len(fleet.items)
    head = f"vllmstat  fleet · {n} instance{'' if n == 1 else 's'}  up {uptime}  {interval:.1f}s"
    cols = f" {'NAME':<14} {'ST':<2} {'RUN/WAIT':>9} {'GEN t/s':>8} {'KV%':>5} {'p50':>7}"
    if show_gpu:
        cols += "  GPU"
    lines = [head, cols]
    for i, (inst, s) in enumerate(fleet.items):
        cur = "▸" if i == selected else " "
        st = "●" if s.connected else "✗"
        rw = f"{s.running:.0f}/{s.waiting:.0f}" if s.connected else "—"
        gen = fmt_si(s.gen_tps) if s.connected else "—"
        kv = fmt_pct(s.kv_usage) if s.connected else "—"
        p50 = fmt_dur(s.ttft.p50) if (s.connected and s.ttft.p50 is not None) else "—"
        row = f"{cur}{inst.name:<14.14} {st:<2} {rw:>9} {gen:>8} {kv:>5} {p50:>7}"
        if show_gpu:
            row += f"  {_gpu_cell(s, inst)}"
        lines.append(row)
    return "\n".join(lines)


def detail_header(inst: Instance, s: Snapshot, *, interval: float, uptime: str) -> str:
    state = "● connected" if s.connected else "✗ down"
    return f"‹ fleet / {inst.name} @ {inst.url}   esc back    {state}  up {uptime}  {interval:.1f}s"


def efficiency(s: Snapshot) -> str:
    pw = sum(g.power_w for g in s.gpu.gpus if g.power_w)
    has_tokw = pw > 0 and s.gen_tps > 0
    if not s.eff_active and not has_tokw:
        return ""
    parts = []
    if s.gflops is not None:
        parts.append(f"{s.gflops:.0f} GFLOP/s")
    if s.gbps is not None:
        parts.append(f"{s.gbps:.0f} GB/s")
    if s.mfu is not None:
        parts.append(f"MFU {fmt_pct(s.mfu)}")
    if has_tokw:
        parts.append(f"{s.gen_tps / pw:.1f} tok/W")
        parts.append(f"{pw / s.gen_tps:.2f} J/tok")
    return "EFFICIENCY  " + " · ".join(parts) if parts else ""


def _tee_one_line(text: str, width: int) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= width else s[: max(1, width - 1)] + "…"


def tee(
    events: list[TeeEvent],
    *,
    width: int | None = None,
    height: int = 10,
    source_desc: str = "",
) -> str:
    w = width or _DEFAULT_PLOT_WIDTH
    head = f"TEE · {source_desc or '—'}"
    rows: list[str] = []
    for e in events:
        if e.kind == "http":
            ts = time.strftime("%H:%M:%S", time.localtime(e.ts))
            mark = "!" if (e.status or 0) >= 400 else " "
            line = f"{mark}{ts} {e.method or '?':<4} {e.path or ''}  {e.status or ''}"
            rows.append(_tee_one_line(line, w))
        elif e.kind == "exchange":
            rows.append(_tee_one_line(f"▶ {e.prompt or ''}", w))
            rows.append(_tee_one_line(f"◀ {e.response or ''}{'' if e.done else ' ▌'}", w))
        else:  # note
            rows.append(_tee_one_line(f"· {e.text or ''}", w))
    if not rows:
        return f"{head}\n (waiting for requests…)"
    visible = rows[-(height - 1) :] if height > 1 else rows[-1:]
    return head + "\n" + "\n".join(visible)


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
        # Clocks: NVIDIA reports both SM and memory; Intel xe has no mem clock,
        # so show just "clk <sm> MHz" (no trailing "/—") when mem is absent.
        if g.clock_sm_mhz is not None:
            if g.clock_mem_mhz is not None:
                parts.append(f"  clk {g.clock_sm_mhz}/{g.clock_mem_mhz} MHz")
            else:
                parts.append(f"  clk {g.clock_sm_mhz} MHz")
        # Hints: util now usually comes from gtidle (non-root), so only flag
        # what's actually missing. VRAM stays root-gated on Intel; if it's the
        # only thing absent, point at that. If nothing reads at all, send the
        # user to the README's GPU-stats section.
        if g.util_gpu is None and g.mem_used is None:
            parts.append("  (GPU stats: see README)")
        elif g.vendor == "intel" and g.mem_used is None:
            parts.append("  (VRAM needs root)")
        lines.append("".join(parts))
    return "\n".join(lines)
