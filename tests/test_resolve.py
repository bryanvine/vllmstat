from __future__ import annotations

from vllmstat.core.resolve import (
    classify_locality,
    instance_from_dict,
    normalize_url,
    resolve_fleet,
)

LN = {"localhost", "127.0.0.1"}


def test_normalize_url():
    assert normalize_url("LocalHost:8000/") == "http://localhost:8000"
    assert normalize_url("http://h:9/metrics/") == "http://h:9/metrics"


def test_classify_locality():
    assert classify_locality("http://localhost:8000", LN) == "local"
    assert classify_locality("http://gpu-box:8000", LN) == "remote"


def test_instance_from_dict_gpus_and_name():
    i = instance_from_dict({"url": "http://localhost:8000", "gpus": [0, 1]}, local_names=LN)
    assert i.gpus == (0, 1) and i.name == "localhost:8000" and i.locality == "local"


def test_instance_from_dict_local_override():
    i = instance_from_dict({"url": "http://gpu-box:8000", "local": True}, local_names=LN)
    assert i.locality == "local"


def test_resolve_fleet_merges_and_dedupes():
    cfg = [instance_from_dict({"url": "http://localhost:8000", "name": "a"}, local_names=LN)]
    fleet = resolve_fleet(cfg, [], ["http://localhost:8000", "http://b:8001"], local_names=LN)
    # dup dropped, order kept
    assert [i.url for i in fleet] == ["http://localhost:8000", "http://b:8001"]


def test_resolve_fleet_default_when_empty():
    fleet = resolve_fleet([], [], [], local_names=LN)
    assert len(fleet) == 1 and fleet[0].url == "http://localhost:8000"


def test_resolve_fleet_dedupes_names():
    a = instance_from_dict({"url": "http://a:8000", "name": "x"}, local_names=LN)
    b = instance_from_dict({"url": "http://b:8000", "name": "x"}, local_names=LN)
    out = resolve_fleet([a, b], [], [], local_names=LN)
    assert [i.name for i in out] == ["x", "x#2"]
