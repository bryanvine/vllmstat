from __future__ import annotations

from dataclasses import asdict

from vllmtop.core.state import Snapshot


def snapshot_to_dict(s: Snapshot) -> dict:
    d = asdict(s)
    # group kv fields for readability
    d["kv"] = {
        "dtype": s.kv_dtype, "capacity_tokens": s.kv_capacity_tokens,
        "used_tokens": s.kv_used_tokens, "ratio": s.kv_ratio,
        "ratio_kind": s.kv_ratio_kind, "fp16_equiv_tokens": s.kv_fp16_equiv_tokens,
        "fp16_full_ctx_gb": s.kv_fp16_full_ctx_gb, "usage": s.kv_usage,
    }
    return d
