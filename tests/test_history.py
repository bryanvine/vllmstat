from vllmtop.core.history import Series, History


def test_series_caps_length():
    s = Series(maxlen=3)
    for i in range(5):
        s.push(float(i))
    assert list(s.values) == [2.0, 3.0, 4.0]


def test_history_named_series_autocreate():
    h = History(maxlen=4)
    h.push("gen_tps", 1.0)
    h.push("gen_tps", 2.0)
    assert list(h.series("gen_tps").values) == [1.0, 2.0]
    assert list(h.series("never_pushed").values) == []
