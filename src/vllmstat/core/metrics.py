from __future__ import annotations

from dataclasses import dataclass

from vllmstat.core.histogram import (
    histogram_fraction_below,
    histogram_max_le,
    histogram_quantile,
    windowed_buckets,
)
from vllmstat.core.kv import compute_kv
from vllmstat.core.parse import (
    Families,
    first_value,
    get_buckets,
    hist_count,
    hist_sum,
    info_labels,
    sum_value,
)
from vllmstat.core.rates import Rate
from vllmstat.core.state import Quantiles, Snapshot

_LAT = {
    "ttft": "vllm:time_to_first_token_seconds",
    "tpot": "vllm:request_time_per_output_token_seconds",
    "e2e": "vllm:e2e_request_latency_seconds",
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
}

_PROMPT_LEN = "vllm:request_prompt_tokens"
_GEN_LEN = "vllm:request_generation_tokens"
_SUCCESS = "vllm:request_success_total"


def _int(s: str | None) -> int | None:
    try:
        return int(s) if s not in (None, "None", "") else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _SessionStats:
    active_s: float = 0.0
    idle_s: float = 0.0
    active_frac: float | None = None
    avg_decode_tps: float | None = None
    avg_prefill_tps: float | None = None
    requests: int = 0
    gen_tokens: float = 0.0
    prompt_tokens: float = 0.0
    avg_gen_tokens_per_req: float | None = None


class MetricsEngine:
    def __init__(
        self,
        *,
        alpha: float = 0.3,
        dims: dict[str, int] | None = None,
        max_model_len: int | None = None,
        ttft_slo_s: float = 1.0,
        tpot_slo_s: float = 0.05,
    ) -> None:
        self.ttft_slo_s = ttft_slo_s
        self.tpot_slo_s = tpot_slo_s
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
        # Session-peak concurrency (high-water marks since the last reset).
        self._peak_running = 0.0
        self._peak_waiting = 0.0
        # Session accumulators (averages while the server is actively serving).
        self._sess_t_prev: float | None = None
        self._sess_active_s = 0.0
        self._sess_idle_s = 0.0
        self._sess_acc_gen = 0.0
        self._sess_acc_prompt = 0.0
        self._sess_gen0: float | None = None
        self._sess_prompt0: float | None = None
        self._sess_req0: float | None = None
        self._sess_prev_gen: float | None = None
        self._sess_prev_prompt: float | None = None

    def reset_session(self) -> None:
        """Clear all session accumulators; the next sample re-baselines."""
        self._peak_running = 0.0
        self._peak_waiting = 0.0
        self._sess_t_prev = None
        self._sess_active_s = 0.0
        self._sess_idle_s = 0.0
        self._sess_acc_gen = 0.0
        self._sess_acc_prompt = 0.0
        self._sess_gen0 = None
        self._sess_prompt0 = None
        self._sess_req0 = None
        self._sess_prev_gen = None
        self._sess_prev_prompt = None

    def _session(
        self,
        running: float,
        gen_total: float,
        prompt_total: float,
        req_total: float,
        now: float,
    ) -> _SessionStats:
        """Accumulate active/idle time and decode/prefill tokens while serving.

        "Active" is a sample with ``running > 0``: its window adds to active
        time and accumulates the generation/prompt-token deltas. Idle windows
        only add to idle time. ``max(0, …)`` on the deltas guards counter
        resets; a drop in ``gen_total`` below the session baseline means the
        server restarted, so we re-baseline and zero the accumulators.
        """
        if self._sess_gen0 is None:
            self._sess_gen0 = gen_total
            self._sess_prompt0 = prompt_total
            self._sess_req0 = req_total
            self._sess_prev_gen = gen_total
            self._sess_prev_prompt = prompt_total
            self._sess_t_prev = now
            return _SessionStats()

        if gen_total < self._sess_gen0:
            # Counter reset (server restart): re-baseline, drop accumulators.
            self._sess_gen0 = gen_total
            self._sess_prompt0 = prompt_total
            self._sess_req0 = req_total
            self._sess_prev_gen = gen_total
            self._sess_prev_prompt = prompt_total
            self._sess_t_prev = now
            self._sess_active_s = 0.0
            self._sess_idle_s = 0.0
            self._sess_acc_gen = 0.0
            self._sess_acc_prompt = 0.0
            return _SessionStats()

        assert self._sess_t_prev is not None
        assert self._sess_prev_gen is not None
        assert self._sess_prev_prompt is not None
        dt = now - self._sess_t_prev
        if dt > 0:
            dgen = max(0.0, gen_total - self._sess_prev_gen)
            dprompt = max(0.0, prompt_total - self._sess_prev_prompt)
            if running > 0:
                self._sess_active_s += dt
                self._sess_acc_gen += dgen
                self._sess_acc_prompt += dprompt
            else:
                self._sess_idle_s += dt
            self._sess_prev_gen = gen_total
            self._sess_prev_prompt = prompt_total
            self._sess_t_prev = now

        active_s = self._sess_active_s
        total_s = active_s + self._sess_idle_s
        gen0 = self._sess_gen0 or 0.0
        prompt0 = self._sess_prompt0 or 0.0
        gen_tokens = max(0.0, gen_total - gen0)
        prompt_tokens = max(0.0, prompt_total - prompt0)
        requests = int(req_total - self._sess_req0) if self._sess_req0 is not None else 0
        return _SessionStats(
            active_s=active_s,
            idle_s=self._sess_idle_s,
            active_frac=(active_s / total_s) if total_s > 0 else None,
            avg_decode_tps=(self._sess_acc_gen / active_s) if active_s > 0 else None,
            avg_prefill_tps=(self._sess_acc_prompt / active_s) if active_s > 0 else None,
            requests=requests,
            gen_tokens=gen_tokens,
            prompt_tokens=prompt_tokens,
            avg_gen_tokens_per_req=(gen_tokens / requests) if requests > 0 else None,
        )

    def _window_buckets(self, fam: Families, base: str) -> list[tuple[float, float]]:
        cur = get_buckets(fam, base)
        if not cur:
            return []
        if self._prev is not None:
            prev = get_buckets(self._prev, base)
            if prev:
                delta = windowed_buckets(prev, cur)
                if delta and delta[-1][1] > 0:
                    return delta
        return cur

    def _quantiles(self, fam: Families, base: str) -> Quantiles:
        buckets = self._window_buckets(fam, base)
        if not buckets:
            return Quantiles()
        return Quantiles(
            p50=histogram_quantile(buckets, 0.50),
            p90=histogram_quantile(buckets, 0.90),
            p99=histogram_quantile(buckets, 0.99),
        )

    def _hist_stats(self, fam: Families, base: str) -> Quantiles:
        q = self._quantiles(fam, base)
        cnt, s = hist_count(fam, base), hist_sum(fam, base)
        mean: float | None = None
        if self._prev is not None and cnt is not None and s is not None:
            pc, ps = hist_count(self._prev, base), hist_sum(self._prev, base)
            if pc is not None and ps is not None and cnt > pc:
                mean = (s - ps) / (cnt - pc)
        if mean is None and cnt and s is not None and cnt > 0:
            mean = s / cnt
        return Quantiles(p50=q.p50, p90=q.p90, p99=q.p99, mean=mean)

    def _finish_reasons(self, fam: Families) -> dict[str, float]:
        cur = {lbl.get("finished_reason", "?"): v for lbl, v in fam.get(_SUCCESS, [])}
        if not cur:
            return {}
        base = cur
        if self._prev is not None:
            prev = {lbl.get("finished_reason", "?"): v for lbl, v in self._prev.get(_SUCCESS, [])}
            delta = {k: max(0.0, cur.get(k, 0.0) - prev.get(k, 0.0)) for k in cur}
            if sum(delta.values()) > 0:
                base = delta
        total = sum(base.values())
        return {k: v / total for k, v in base.items()} if total > 0 else {}

    def derive(self, fam: Families, now: float) -> Snapshot:
        labels = info_labels(fam, "vllm:cache_config_info")
        model_names = sorted(
            {lbl.get("model_name", "") for lbl, _ in fam.get("vllm:num_requests_running", [])}
            - {""}
        )
        engines = {lbl.get("engine") for lbl, _ in fam.get("vllm:num_requests_running", [])}

        running = sum_value(fam, "vllm:num_requests_running") or 0.0
        waiting = sum_value(fam, "vllm:num_requests_waiting") or 0.0
        self._peak_running = max(self._peak_running, running)
        self._peak_waiting = max(self._peak_waiting, waiting)

        # max context seen (bucketed lifetime maxes; tune --max-model-len)
        max_prompt = histogram_max_le(get_buckets(fam, _PROMPT_LEN))
        max_output = histogram_max_le(get_buckets(fam, _GEN_LEN))

        # throughput rates
        gen_total = sum_value(fam, "vllm:generation_tokens_total") or 0.0
        prompt_total = sum_value(fam, "vllm:prompt_tokens_total") or 0.0
        req_total = sum_value(fam, "vllm:request_success_total") or 0.0
        gen = self._gen.update(gen_total, now)
        prompt = self._prompt.update(prompt_total, now)
        req = self._req.update(req_total, now)
        preempt = self._preempt.update(sum_value(fam, "vllm:num_preemptions_total") or 0.0, now)

        sess = self._session(running, gen_total, prompt_total, req_total, now)

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
        spec_acceptance = (
            (accepted / draft_tokens)
            if (spec_active and accepted is not None and draft_tokens is not None)
            else None
        )
        spec_per_draft = (
            (accepted / drafts) if (spec_active and drafts and accepted is not None) else None
        )

        # efficiency (conditional)
        flops = self._flops.update(sum_value(fam, "vllm:estimated_flops_per_gpu_total") or 0.0, now)
        rbytes = self._rbytes.update(
            sum_value(fam, "vllm:estimated_read_bytes_per_gpu_total") or 0.0, now
        )
        wbytes = self._wbytes.update(
            sum_value(fam, "vllm:estimated_write_bytes_per_gpu_total") or 0.0, now
        )
        eff_active = (flops > 0) or (rbytes + wbytes > 0)

        snap = Snapshot(
            ts=now,
            connected=True,
            model_names=model_names or ([mn] if (mn := labels.get("model_name")) else []),
            engine_count=len([e for e in engines if e is not None]) or 1,
            max_model_len=self.max_model_len,
            running=running,
            waiting=waiting,
            peak_running=self._peak_running,
            peak_waiting=self._peak_waiting,
            preempt_rate=preempt,
            gen_tps=gen,
            prompt_tps=prompt,
            req_rate=req,
            tokens_per_iter=tokens_per_iter,
            session_active_s=sess.active_s,
            session_idle_s=sess.idle_s,
            session_active_frac=sess.active_frac,
            avg_decode_tps=sess.avg_decode_tps,
            avg_prefill_tps=sess.avg_prefill_tps,
            session_requests=sess.requests,
            session_gen_tokens=sess.gen_tokens,
            session_prompt_tokens=sess.prompt_tokens,
            avg_gen_tokens_per_req=sess.avg_gen_tokens_per_req,
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
            prefill=self._quantiles(fam, _LAT["prefill"]),
            decode=self._quantiles(fam, _LAT["decode"]),
            prompt_len=self._hist_stats(fam, _PROMPT_LEN),
            gen_len=self._hist_stats(fam, _GEN_LEN),
            max_prompt_tokens=int(max_prompt) if max_prompt is not None else None,
            max_output_tokens=int(max_output) if max_output is not None else None,
            finish_reasons=self._finish_reasons(fam),
            goodput_ttft=histogram_fraction_below(
                self._window_buckets(fam, _LAT["ttft"]), self.ttft_slo_s
            ),
            goodput_tpot=histogram_fraction_below(
                self._window_buckets(fam, _LAT["tpot"]), self.tpot_slo_s
            ),
            ttft_slo_s=self.ttft_slo_s,
            tpot_slo_s=self.tpot_slo_s,
            spec_active=spec_active,
            spec_acceptance=spec_acceptance,
            spec_accepted_per_draft=spec_per_draft,
            eff_active=eff_active,
            gflops=(flops / 1e9) if eff_active else None,
            gbps=((rbytes + wbytes) / 1e9) if eff_active else None,
        )
        self._prev = fam
        return snap
