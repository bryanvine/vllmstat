import math

from vllmtop.core.rates import Rate


def test_first_update_is_zero():
    r = Rate(alpha=0.5)
    assert r.update(100.0, t=0.0) == 0.0


def test_steady_rate_converges():
    r = Rate(alpha=0.5)
    r.update(0.0, t=0.0)
    v = 0.0
    val = 0.0
    for i in range(1, 50):
        v += 10.0
        val = r.update(v, t=float(i))  # +10 per 1s -> 10/s
    assert math.isclose(val, 10.0, rel_tol=1e-3)


def test_counter_reset_does_not_spike():
    r = Rate(alpha=0.5)
    r.update(1000.0, t=0.0)
    r.update(1010.0, t=1.0)
    before = r.value
    after = r.update(5.0, t=2.0)  # reset to a small value
    assert after == before  # no negative/huge spike, value held


def test_zero_dt_held():
    r = Rate(alpha=0.5)
    r.update(0.0, t=1.0)
    r.update(10.0, t=1.0)  # dt=0
    assert r.value == 0.0
