from __future__ import annotations

from vllmtop.core.histogram import histogram_quantile, windowed_buckets
from vllmtop.core.kv import compute_kv
from vllmtop.core.parse import (
    Families, first_value, get_buckets, hist_count, info_labels, sum_value,
)
from vllmtop.core.rates import Rate
from vllmtop.core.state import Quantiles, Snapshot

_LAT = {
    "ttft": "vllm:time_to_first_token_seconds",
    "tpot": "vllm:request_time_per_output_token_seconds",
    "e2e": "vllm:e2e_request_latency_seconds",
    "queue": "vllm:request_queue_time_seconds",
}


def _int(s: str | None) -> int | None:
    try:
        return int(s) if s not in (None, "None", "") else None
    except (TypeError, ValueError):
        return None


class MetricsEngine:
    def __init__(
        self, *, alpha: float = 0.3, dims: dict[str, int] | None = None,
        max_model_len: int | None = None,
    ) -> None:
        self.dims = dims
        self.max_model_len = max_model_len
        self._gen = Rate(alpha)
        self._prompt = Rate(alpha)
        self._req = Rate(alpha)
        self._preempt = Rate(alpha)
        self._flops = Rate(alpha)
        self._rbytes = Rate(alpha)
        self._wbytes = Rate(alpha)
        self._prev: Families | None = None

    def _quantiles(self, fam: Families, base: str) -> Quantiles:
        cur = get_buckets(fam, base)
        if not cur:
            return Quantiles()
        buckets = cur
        if self._prev is not None:
            prev = get_buckets(self._prev, base)
            if prev:
                delta = windowed_buckets(prev, cur)
                if delta and delta[-1][1] > 0:
                    buckets = delta
        return Quantiles(
            p50=histogram_quantile(buckets, 0.50),
            p90=histogram_quantile(buckets, 0.90),
            p99=histogram_quantile(buckets, 0.99),
        )

    def derive(self, fam: Families, now: float) -> Snapshot:
        labels = info_labels(fam, "vllm:cache_config_info")
        model_names = sorted({
            lbl.get("model_name", "") for lbl, _ in fam.get("vllm:num_requests_running", [])
        } - {""})
        engines = {lbl.get("engine") for lbl, _ in fam.get("vllm:num_requests_running", [])}

        # throughput rates
        gen = self._gen.update(sum_value(fam, "vllm:generation_tokens_total") or 0.0, now)
        prompt = self._prompt.update(sum_value(fam, "vllm:prompt_tokens_total") or 0.0, now)
        req = self._req.update(sum_value(fam, "vllm:request_success_total") or 0.0, now)
        preempt = self._preempt.update(sum_value(fam, "vllm:num_preemptions_total") or 0.0, now)

        # tokens/iter mean
        it_sum = sum_value(fam, "vllm:iteration_tokens_total_sum")
        it_cnt = sum_value(fam, "vllm:iteration_tokens_total_count")
        tokens_per_iter = (it_sum / it_cnt) if (it_sum and it_cnt) else None

        # cache reuse
        q = sum_value(fam, "vllm:prefix_cache_queries_total") or 0.0
        h = sum_value(fam, "vllm:prefix_cache_hits_total") or 0.0
        hit_life = (h / q) if q > 0 else None
        hit_win = None
        if self._prev is not None:
            pq = sum_value(self._prev, "vllm:prefix_cache_queries_total") or 0.0
            ph = sum_value(self._prev, "vllm:prefix_cache_hits_total") or 0.0
            dq, dh = q - pq, h - ph
            if dq > 0:
                hit_win = max(0.0, min(1.0, dh / dq))

        # token sources
        src = {lbl.get("source"): v for lbl, v in fam.get("vllm:prompt_tokens_by_source_total", [])}
        src_total = sum(src.values()) or 0.0
        frac = lambda k: (src.get(k, 0.0) / src_total) if src_total > 0 else None  # noqa: E731

        ext_q = sum_value(fam, "vllm:external_prefix_cache_queries_total") or 0.0
        external_active = ext_q > 0 or (src.get("external_kv_transfer", 0.0) > 0)

        # kv memory
        kv_usage = first_value(fam, "vllm:kv_cache_usage_perc") or 0.0
        kv = compute_kv(
            cache_dtype=labels.get("cache_dtype"),
            num_gpu_blocks=_int(labels.get("num_gpu_blocks")),
            block_size=_int(labels.get("block_size")),
            kv_usage=kv_usage,
            kv_cache_memory_bytes=_int(labels.get("kv_cache_memory_bytes")),
            dims=self.dims,
            max_model_len=self.max_model_len,
        )

        # spec decode
        drafts = sum_value(fam, "vllm:spec_decode_num_drafts_total")
        draft_tokens = sum_value(fam, "vllm:spec_decode_num_draft_tokens_total")
        accepted = sum_value(fam, "vllm:spec_decode_num_accepted_tokens_total")
        spec_active = bool(drafts and draft_tokens)
        spec_acceptance = (accepted / draft_tokens) if (spec_active and accepted is not None) else None
        spec_per_draft = (accepted / drafts) if (spec_active and drafts and accepted is not None) else None

        # efficiency (conditional)
        flops = self._flops.update(sum_value(fam, "vllm:estimated_flops_per_gpu_total") or 0.0, now)
        rbytes = self._rbytes.update(sum_value(fam, "vllm:estimated_read_bytes_per_gpu_total") or 0.0, now)
        wbytes = self._wbytes.update(sum_value(fam, "vllm:estimated_write_bytes_per_gpu_total") or 0.0, now)
        eff_active = (flops > 0) or (rbytes + wbytes > 0)

        snap = Snapshot(
            ts=now,
            connected=True,
            model_names=model_names or ([labels.get("model_name")] if labels.get("model_name") else []),
            engine_count=len([e for e in engines if e is not None]) or 1,
            running=sum_value(fam, "vllm:num_requests_running") or 0.0,
            waiting=sum_value(fam, "vllm:num_requests_waiting") or 0.0,
            preempt_rate=preempt,
            gen_tps=gen,
            prompt_tps=prompt,
            req_rate=req,
            tokens_per_iter=tokens_per_iter,
            prefix_hit_window=hit_win,
            prefix_hit_lifetime=hit_life,
            src_compute=frac("local_compute"),
            src_cache_hit=frac("local_cache_hit"),
            src_external=frac("external_kv_transfer"),
            cached_tokens_total=sum_value(fam, "vllm:prompt_tokens_cached_total") or 0.0,
            recomputed_tokens_total=sum_value(fam, "vllm:prompt_tokens_recomputed_total") or 0.0,
            external_kv_active=external_active,
            kv_usage=kv_usage,
            kv_capacity_tokens=kv.capacity_tokens,
            kv_used_tokens=kv.used_tokens,
            kv_dtype=kv.dtype,
            kv_ratio=kv.ratio,
            kv_ratio_kind=kv.ratio_kind,
            kv_fp16_equiv_tokens=kv.fp16_equiv_tokens,
            kv_fp16_full_ctx_gb=kv.fp16_full_ctx_gb,
            ttft=self._quantiles(fam, _LAT["ttft"]),
            tpot=self._quantiles(fam, _LAT["tpot"]),
            e2e=self._quantiles(fam, _LAT["e2e"]),
            queue=self._quantiles(fam, _LAT["queue"]),
            spec_active=spec_active,
            spec_acceptance=spec_acceptance,
            spec_accepted_per_draft=spec_per_draft,
            eff_active=eff_active,
            gflops=(flops / 1e9) if eff_active else None,
            gbps=((rbytes + wbytes) / 1e9) if eff_active else None,
        )
        self._prev = fam
        return snap
