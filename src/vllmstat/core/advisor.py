from __future__ import annotations

from dataclasses import dataclass

from vllmstat.core.state import Snapshot

# Thresholds, kept as named constants so the rules below read declaratively.
_PREEMPT_MIN_RATE = 0.05  # preemptions/s above which KV pressure is real (not an EWMA residual)
_TRUNC_ERROR = 0.50  # length-finish fraction at/above which completions are clearly cut off
_TRUNC_WARN = 0.20
_REQUEST_ERROR_FRAC = 0.02  # error finish-reason fraction worth surfacing
_CONTEXT_OVERPROV = 0.50  # max context used below this fraction of max-model-len → wasteful
_GPU_MEM_UTIL_LOW = 0.85  # gpu_memory_utilization below this → under-committed
_QUEUE_BACKLOG_PEAK = 2  # session-peak waiting at/above this → a real backlog formed
_SPEC_ACCEPT_LOW = 0.30  # spec-decode acceptance below this → mostly overhead
_GPU_TEMP_HOT_C = 85.0
_GPU_POWER_FRAC = 0.98  # power draw within this fraction of the limit → throttle risk


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" | "warn"
    code: str  # stable identifier (for tests / filtering)
    message: str


def _pct(frac: float) -> str:
    return f"{frac * 100:.0f}%"


def detect_issues(s: Snapshot) -> list[Issue]:
    """Inspect a Snapshot for misconfigurations and missed optimizations.

    Pure function returning issues sorted error-first, then by rule order. Each
    rule fires only on a clear signal so the panel stays quiet unless something
    is genuinely worth acting on. Returns ``[]`` when the server looks healthy.
    """
    errors: list[Issue] = []
    warns: list[Issue] = []

    # 1. KV cache exhausted — active preemptions (context OOM in practice).
    if s.preempt_rate >= _PREEMPT_MIN_RATE:
        errors.append(
            Issue(
                "error",
                "kv_preemption",
                f"KV cache exhausted — {s.preempt_rate:.1f} preemptions/s; running requests are "
                "being evicted and recomputed. Lower --max-model-len/--max-num-seqs or raise "
                "--gpu-memory-utilization.",
            )
        )

    # 2 / 4. Output truncation at the length cap (silent context loss).
    length_frac = s.finish_reasons.get("length", 0.0)
    if length_frac >= _TRUNC_ERROR:
        errors.append(
            Issue(
                "error",
                "output_truncation",
                f"{_pct(length_frac)} of requests truncated at the length cap — completions are "
                "being cut off. Raise max_tokens or --max-model-len.",
            )
        )
    elif length_frac >= _TRUNC_WARN:
        warns.append(
            Issue(
                "warn",
                "output_truncation",
                f"{_pct(length_frac)} of requests hit the length cap — some completions "
                "truncated. Consider a higher max_tokens.",
            )
        )

    # 3. Request errors.
    err_frac = s.finish_reasons.get("error", 0.0)
    if err_frac >= _REQUEST_ERROR_FRAC:
        errors.append(
            Issue(
                "error",
                "request_errors",
                f"{_pct(err_frac)} of requests ended in error — check the server logs.",
            )
        )

    # 5. Context over-provisioned — largest request ≪ max-model-len (wasted KV cache).
    if s.max_model_len and s.max_prompt_tokens is not None and s.max_output_tokens is not None:
        total = s.max_prompt_tokens + s.max_output_tokens
        if total / s.max_model_len < _CONTEXT_OVERPROV:
            warns.append(
                Issue(
                    "warn",
                    "context_overprovisioned",
                    f"Context over-provisioned — largest request {total:,} of "
                    f"{s.max_model_len:,} max-model-len ({_pct(total / s.max_model_len)}). "
                    "Shrink --max-model-len to free KV cache for more concurrency.",
                )
            )

    # 6. GPU memory under-committed.
    if s.gpu_memory_utilization is not None and s.gpu_memory_utilization < _GPU_MEM_UTIL_LOW:
        warns.append(
            Issue(
                "warn",
                "gpu_mem_under",
                f"GPU memory under-committed — gpu_memory_utilization="
                f"{s.gpu_memory_utilization:.2f}; raising it gives vLLM more KV cache "
                "(more concurrency / longer context).",
            )
        )

    # 7. Queue backlog — a real waiting queue formed this session.
    if s.peak_waiting >= _QUEUE_BACKLOG_PEAK:
        warns.append(
            Issue(
                "warn",
                "queue_backlog",
                f"Queue backlog — peak {s.peak_waiting:.0f} requests waiting. Raise "
                "--max-num-seqs or add a replica.",
            )
        )

    # 8. Spec-decode enabled but ineffective.
    if s.spec_active and s.spec_acceptance is not None and s.spec_acceptance < _SPEC_ACCEPT_LOW:
        warns.append(
            Issue(
                "warn",
                "spec_ineffective",
                f"Speculative decoding acceptance is low ({_pct(s.spec_acceptance)}) — most "
                "drafted tokens are rejected, adding overhead. Try a better draft model or "
                "disable it.",
            )
        )

    # 9. Prefix caching disabled.
    if s.prefix_caching_enabled is False:
        warns.append(
            Issue(
                "warn",
                "prefix_caching_off",
                "Prefix caching is disabled — enabling it speeds up shared-prefix workloads "
                "(system prompts, few-shot, multi-turn chats).",
            )
        )

    # 10. GPU near thermal / power limit.
    for g in s.gpu.gpus:
        hot = g.temp_c is not None and g.temp_c >= _GPU_TEMP_HOT_C
        near_power = bool(
            g.power_w and g.power_limit_w and g.power_w >= _GPU_POWER_FRAC * g.power_limit_w
        )
        if not (hot or near_power):
            continue
        temp = f"{g.temp_c:.0f}°C" if g.temp_c is not None else "—"
        power = (
            f"{g.power_w:.0f}W of {g.power_limit_w:.0f}W"
            if (g.power_w and g.power_limit_w)
            else "—"
        )
        warns.append(
            Issue(
                "warn",
                "gpu_thermal",
                f"GPU {g.index} near limit — {temp} / {power}; possible throttling.",
            )
        )

    return errors + warns
