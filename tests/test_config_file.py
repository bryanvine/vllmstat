from __future__ import annotations

import pytest

from vllmstat.core.config_file import find_config, parse_config


def test_parse_config_instances_and_globals():
    insts, g = parse_config(
        'interval = 2.0\n[[instance]]\nname="a"\nurl="http://a:8000"\ngpus=[0]\n'
    )
    assert g["interval"] == 2.0
    assert insts == [{"name": "a", "url": "http://a:8000", "gpus": [0]}]


def test_parse_config_bad_toml_raises():
    with pytest.raises(ValueError):
        parse_config("this is = = not toml")


def test_parse_config_instance_must_be_array():
    with pytest.raises(ValueError):
        parse_config('instance = "oops"')


def test_find_config_precedence():
    seen = {"/explicit.toml", "./vllmstat.toml"}
    assert find_config("/explicit.toml", {}, exists=lambda p: p in seen) == "/explicit.toml"
    assert (
        find_config(None, {"VLLMSTAT_CONFIG": "/env.toml"}, exists=lambda p: False) == "/env.toml"
    )  # noqa: E501
    assert find_config(None, {}, exists=lambda p: p in seen) == "./vllmstat.toml"
    assert find_config(None, {}, exists=lambda p: False) is None
