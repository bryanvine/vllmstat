from __future__ import annotations

from vllmtop.core.history import History
from vllmtop.core.state import Snapshot
from vllmtop.format import fmt_bytes, fmt_dur, fmt_pct, fmt_si, sparkline


def header(s: Snapshot, *, url: str, interval: float, uptime: str) -> str:
    state = "● connected" if s.connected else "● down"
    models = ",".join(s.model_names) or "—"
    parts = f"vllmtop  {models} @ {url}  engines {s.engine_count}"
    return f"{parts}  {state}  up {uptime}  {interval:.1f}s"


def concurrency(s: Snapshot, h: History) -> str:
    run_spark = sparkline(list(h.series("running").values))
    wait_spark = sparkline(list(h.series("waiting").values))
    seqs = f"  max-seqs {s.max_num_seqs}" if s.max_num_seqs else ""
    return (
        f"CONCURRENCY\n"
        f" running {s.running:.0f}  ▕{run_spark}▏\n"
        f" waiting {s.waiting:.0f}  ▕{wait_spark}▏\n"
        f" preempt {s.preempt_rate:.1f}/s{seqs}"
    )


def throughput(s: Snapshot, h: History) -> str:
    g = sparkline(list(h.series("gen_tps").values))
    p = sparkline(list(h.series("prompt_tps").values))
    tpi = f"{s.tokens_per_iter:.0f}" if s.tokens_per_iter else "—"
    return (
        f"THROUGHPUT\n"
        f" gen    {s.gen_tps:.0f} tok/s ▕{g}▏\n"
        f" prompt {s.prompt_tps:.0f} tok/s ▕{p}▏\n"
        f" tok/iter {tpi} · {s.req_rate:.1f} req/s"
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
    return (
        f"SPEC DECODE  acceptance {fmt_pct(s.spec_acceptance)}  "
        f"accepted/draft {s.spec_accepted_per_draft:.2f}"
    )


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
        mem_pct = (g.mem_used / g.mem_total) if (g.mem_used and g.mem_total) else None
        lines.append(
            f"GPU {g.index}  {g.name}  {g.util_gpu:.0f}%  "
            f"{fmt_bytes(g.mem_used)}/{fmt_bytes(g.mem_total)} ({fmt_pct(mem_pct)})  "
            f"{g.temp_c:.0f}°C  {g.power_w:.0f}/{g.power_limit_w:.0f} W  "
            f"clk {g.clock_sm_mhz}/{g.clock_mem_mhz} MHz"
        )
    return "\n".join(lines)


def footer(s: Snapshot) -> str:
    gpu_badge = "GPU ✓" if s.gpu.available else "GPU ✗"
    vllm_badge = "vLLM ✓" if s.connected else "vLLM ✗"
    return f"[q]uit [p]ause [+/-]interval [g]pu [?]help    ● {vllm_badge}  {gpu_badge}"
