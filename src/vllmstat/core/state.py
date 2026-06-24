from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Quantiles:
    p50: float | None = None
    p90: float | None = None
    p99: float | None = None
    mean: float | None = None


@dataclass
class GpuSample:
    index: int
    name: str
    vendor: str = ""  # "nvidia" | "amd" | "intel" | "other"
    util_gpu: float | None = None  # percent 0-100
    mem_used: int | None = None  # bytes
    mem_total: int | None = None  # bytes
    temp_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    fan_pct: float | None = None  # NVIDIA reports fan as a percentage
    fan_rpm: int | None = None  # AMD/Intel hwmon report fan as RPM
    clock_sm_mhz: int | None = None
    clock_mem_mhz: int | None = None


@dataclass
class GpuSnapshot:
    available: bool = False
    source: str = "none"  # "nvml" | "nvidia-smi" | "none"
    gpus: list[GpuSample] = field(default_factory=list)
    error: str | None = None


@dataclass
class Instance:
    name: str
    url: str
    metrics_path: str = "/metrics"
    api_key: str | None = None
    gpus: tuple[int, ...] = ()
    locality: str = "local"  # "local" | "remote"
    logs: str | None = None  # tee log source: "docker:NAME" or a file path


@dataclass
class FleetSnapshot:
    ts: float
    items: list[tuple[Instance, Snapshot]] = field(default_factory=list)
    gpu: GpuSnapshot = field(default_factory=GpuSnapshot)


@dataclass
class Snapshot:
    ts: float
    connected: bool
    error: str | None = None
    model_names: list[str] = field(default_factory=list)
    engine_count: int = 0
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    # server config (from vllm:cache_config_info) — for the config advisor
    gpu_memory_utilization: float | None = None
    prefix_caching_enabled: bool | None = None
    # concurrency
    running: float = 0.0
    waiting: float = 0.0
    peak_running: float = 0.0  # session-peak num_requests_running
    peak_waiting: float = 0.0  # session-peak num_requests_waiting
    preempt_rate: float = 0.0
    # throughput
    gen_tps: float = 0.0
    prompt_tps: float = 0.0
    req_rate: float = 0.0
    tokens_per_iter: float | None = None
    # session averages (accumulated while the server is actively serving)
    session_active_s: float = 0.0
    session_idle_s: float = 0.0
    session_active_frac: float | None = None
    avg_decode_tps: float | None = None
    avg_prefill_tps: float | None = None
    session_requests: int = 0
    session_gen_tokens: float = 0.0
    session_prompt_tokens: float = 0.0
    avg_gen_tokens_per_req: float | None = None
    # cache reuse
    prefix_hit_window: float | None = None  # 0-1
    prefix_hit_lifetime: float | None = None  # 0-1
    src_compute: float | None = None  # fraction 0-1
    src_cache_hit: float | None = None
    src_external: float | None = None
    cached_tokens_total: float = 0.0
    recomputed_tokens_total: float = 0.0
    external_kv_active: bool = False
    # kv memory / compression
    kv_usage: float = 0.0  # 0-1
    kv_capacity_tokens: int | None = None
    kv_used_tokens: int | None = None
    kv_dtype: str | None = None
    kv_ratio: float | None = None
    kv_ratio_kind: str = "none"  # "achieved" | "nominal" | "none"
    kv_fp16_equiv_tokens: int | None = None
    kv_fp16_full_ctx_gb: float | None = None
    # latency quantiles (seconds)
    ttft: Quantiles = field(default_factory=Quantiles)
    tpot: Quantiles = field(default_factory=Quantiles)
    e2e: Quantiles = field(default_factory=Quantiles)
    queue: Quantiles = field(default_factory=Quantiles)
    # latency phases (seconds)
    prefill: Quantiles = field(default_factory=Quantiles)
    decode: Quantiles = field(default_factory=Quantiles)
    # request shape (token counts; .mean = average)
    prompt_len: Quantiles = field(default_factory=Quantiles)
    gen_len: Quantiles = field(default_factory=Quantiles)
    # max context (bucketed lifetime maxes; for tuning --max-model-len)
    max_prompt_tokens: int | None = None
    max_output_tokens: int | None = None
    # outcomes
    finish_reasons: dict[str, float] = field(default_factory=dict)
    goodput_ttft: float | None = None
    goodput_tpot: float | None = None
    ttft_slo_s: float = 1.0
    tpot_slo_s: float = 0.05
    # speculative decoding
    spec_active: bool = False
    spec_acceptance: float | None = None  # accepted / draft_tokens
    spec_accepted_per_draft: float | None = None
    spec_per_pos: list[float] = field(default_factory=list)
    # efficiency (conditional)
    eff_active: bool = False
    gflops: float | None = None
    gbps: float | None = None
    mfu: float | None = None
    bw_util: float | None = None
    idle_watts_avg: float | None = None  # mean GPU power while idle (running == 0)
    # energy efficiency: session means accumulated only while actively generating, so they
    # persist (frozen) when the server goes idle instead of decaying or blowing up.
    tokens_per_watt: float | None = None
    joules_per_token: float | None = None
    # gpu
    gpu: GpuSnapshot = field(default_factory=GpuSnapshot)


@dataclass
class EnergyView:
    """Read-only energy figures for the TUI (assembled from the store + live data)."""
    available: bool = False
    currency: str = "$"
    today_kwh: float = 0.0
    today_cost: float | None = None
    alltime_kwh: float = 0.0
    alltime_cost: float | None = None
    now_w: float | None = None
    rate: float | None = None
    rate_label: str = ""
