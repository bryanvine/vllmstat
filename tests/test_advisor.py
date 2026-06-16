from vllmstat.core.advisor import detect_issues
from vllmstat.core.state import GpuSample, GpuSnapshot, Snapshot


def _codes(issues) -> set[str]:
    return {i.code for i in issues}


def test_clean_snapshot_no_issues():
    # Healthy server: prefix caching on, memory committed, largest request fills most of
    # the window, no preemption/truncation/queue/errors.
    s = Snapshot(
        ts=0.0,
        connected=True,
        gpu_memory_utilization=0.9,
        prefix_caching_enabled=True,
        max_model_len=8192,
        max_prompt_tokens=3000,
        max_output_tokens=2000,  # 5000/8192 ≈ 61% -> not over-provisioned
    )
    assert detect_issues(s) == []


def test_kv_preemption_is_error():
    issues = detect_issues(Snapshot(ts=0.0, connected=True, preempt_rate=2.3))
    assert any(i.code == "kv_preemption" and i.severity == "error" for i in issues)
    # an EWMA residual below the floor must not fire
    assert "kv_preemption" not in _codes(
        detect_issues(Snapshot(ts=0.0, connected=True, preempt_rate=0.01))
    )


def test_output_truncation_error_and_warn_tiers():
    err = detect_issues(Snapshot(ts=0.0, connected=True, finish_reasons={"length": 0.64}))
    assert any(i.code == "output_truncation" and i.severity == "error" for i in err)
    warn = detect_issues(Snapshot(ts=0.0, connected=True, finish_reasons={"length": 0.30}))
    assert any(i.code == "output_truncation" and i.severity == "warn" for i in warn)
    # low truncation is normal -> no issue
    assert "output_truncation" not in _codes(
        detect_issues(Snapshot(ts=0.0, connected=True, finish_reasons={"length": 0.05}))
    )


def test_request_errors_is_error():
    s = Snapshot(ts=0.0, connected=True, finish_reasons={"error": 0.1})
    assert any(i.code == "request_errors" and i.severity == "error" for i in detect_issues(s))


def test_context_overprovisioned_warn():
    s = Snapshot(
        ts=0.0, connected=True, max_model_len=32768, max_prompt_tokens=500, max_output_tokens=500
    )
    assert "context_overprovisioned" in _codes(detect_issues(s))
    # largest request fills most of the window -> not over-provisioned
    s2 = Snapshot(
        ts=0.0, connected=True, max_model_len=4096, max_prompt_tokens=2000, max_output_tokens=1000
    )
    assert "context_overprovisioned" not in _codes(detect_issues(s2))


def test_context_overprovisioned_needs_request_data():
    # No requests observed yet -> no max_* tokens -> rule cannot fire (no false positive).
    s = Snapshot(ts=0.0, connected=True, max_model_len=32768)
    assert "context_overprovisioned" not in _codes(detect_issues(s))


def test_gpu_mem_under_committed_warn():
    assert "gpu_mem_under" in _codes(
        detect_issues(Snapshot(ts=0.0, connected=True, gpu_memory_utilization=0.70))
    )
    assert "gpu_mem_under" not in _codes(
        detect_issues(Snapshot(ts=0.0, connected=True, gpu_memory_utilization=0.90))
    )


def test_queue_backlog_warn():
    assert "queue_backlog" in _codes(
        detect_issues(Snapshot(ts=0.0, connected=True, peak_waiting=7.0))
    )
    assert "queue_backlog" not in _codes(
        detect_issues(Snapshot(ts=0.0, connected=True, peak_waiting=1.0))
    )


def test_spec_ineffective_warn():
    assert "spec_ineffective" in _codes(
        detect_issues(Snapshot(ts=0.0, connected=True, spec_active=True, spec_acceptance=0.12))
    )
    assert "spec_ineffective" not in _codes(
        detect_issues(Snapshot(ts=0.0, connected=True, spec_active=True, spec_acceptance=0.55))
    )


def test_prefix_caching_off_warn():
    assert "prefix_caching_off" in _codes(
        detect_issues(Snapshot(ts=0.0, connected=True, prefix_caching_enabled=False))
    )
    # unknown (None) must not fire
    assert "prefix_caching_off" not in _codes(
        detect_issues(Snapshot(ts=0.0, connected=True, prefix_caching_enabled=None))
    )


def test_gpu_thermal_warn_on_temp_or_power():
    hot = Snapshot(
        ts=0.0,
        connected=True,
        gpu=GpuSnapshot(available=True, gpus=[GpuSample(index=0, name="x", temp_c=88.0)]),
    )
    assert "gpu_thermal" in _codes(detect_issues(hot))
    near_power = Snapshot(
        ts=0.0,
        connected=True,
        gpu=GpuSnapshot(
            available=True, gpus=[GpuSample(index=0, name="x", power_w=199.0, power_limit_w=200.0)]
        ),
    )
    assert "gpu_thermal" in _codes(detect_issues(near_power))


def test_errors_sorted_before_warnings():
    s = Snapshot(
        ts=0.0,
        connected=True,
        preempt_rate=1.0,  # error
        peak_waiting=5.0,  # warn
        prefix_caching_enabled=False,  # warn
    )
    issues = detect_issues(s)
    assert issues[0].severity == "error"
    assert [i.severity for i in issues] == sorted(
        (i.severity for i in issues), key=lambda x: 0 if x == "error" else 1
    )
