from __future__ import annotations

import re
from dataclasses import dataclass

_FP16 = {"auto", "fp16", "float16", "bf16", "bfloat16", "half"}


def parse_kv_bits(dtype: str | None) -> tuple[int, int] | None:
    if not dtype:
        return None
    d = dtype.lower()
    if d in _FP16:
        return (16, 16)
    if "fp8" in d or "int8" in d or "e4m3" in d or "e5m2" in d:
        return (8, 8)
    m = re.search(r"k(\d+)v(\d+)", d)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def nominal_ratio(dtype: str | None) -> float | None:
    bits = parse_kv_bits(dtype)
    if not bits:
        return None
    bk, bv = bits
    return (16 + 16) / (bk + bv)


def fp16_bytes_per_token(layers: int, kv_heads: int, head_dim: int) -> int:
    return 2 * layers * kv_heads * head_dim * 2


@dataclass
class KvInfo:
    dtype: str | None
    capacity_tokens: int | None
    used_tokens: int | None
    ratio: float | None
    ratio_kind: str            # "achieved" | "nominal" | "none"
    fp16_equiv_tokens: int | None
    fp16_full_ctx_gb: float | None


def compute_kv(
    *,
    cache_dtype: str | None,
    num_gpu_blocks: int | None,
    block_size: int | None,
    kv_usage: float,
    kv_cache_memory_bytes: int | None,
    dims: dict[str, int] | None,
    max_model_len: int | None,
) -> KvInfo:
    capacity = None
    if num_gpu_blocks and block_size:
        capacity = num_gpu_blocks * block_size
    used = round(capacity * kv_usage) if capacity is not None else None

    bpt = None
    if dims and all(k in dims for k in ("layers", "kv_heads", "head_dim")):
        bpt = fp16_bytes_per_token(dims["layers"], dims["kv_heads"], dims["head_dim"])

    ratio: float | None = None
    kind = "none"
    if capacity and bpt and kv_cache_memory_bytes:
        ratio = (capacity * bpt) / kv_cache_memory_bytes
        kind = "achieved"
    else:
        nominal = nominal_ratio(cache_dtype)
        if nominal is not None:
            ratio, kind = nominal, "nominal"

    fp16_equiv = round(capacity / ratio) if (capacity and ratio) else None
    fp16_full_ctx_gb = (max_model_len * bpt / 1e9) if (max_model_len and bpt) else None

    return KvInfo(
        dtype=cache_dtype,
        capacity_tokens=capacity,
        used_tokens=used,
        ratio=ratio,
        ratio_kind=kind,
        fp16_equiv_tokens=fp16_equiv,
        fp16_full_ctx_gb=fp16_full_ctx_gb,
    )
