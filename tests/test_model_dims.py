import json
from pathlib import Path

from vllmtop.model_dims import dims_from_config, load_model_dims

FIX = Path(__file__).parent / "fixtures" / "model_config.json"


def test_dims_from_fixture():
    cfg = json.loads(FIX.read_text())
    assert dims_from_config(cfg) == {"layers": 48, "kv_heads": 4, "head_dim": 128}


def test_dims_head_dim_fallback_from_hidden_size():
    cfg = {"num_hidden_layers": 2, "num_attention_heads": 8, "hidden_size": 1024}
    assert dims_from_config(cfg) == {"layers": 2, "kv_heads": 8, "head_dim": 128}


def test_dims_from_bad_config_returns_none():
    assert dims_from_config({}) is None
    assert dims_from_config({"num_hidden_layers": 1}) is None


def test_load_model_dims_reads_local_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 48, "num_attention_heads": 32,
        "num_key_value_heads": 4, "head_dim": 128,
    }))
    md = load_model_dims(str(tmp_path), 262144)
    assert md.dims == {"layers": 48, "kv_heads": 4, "head_dim": 128}
    assert md.max_model_len == 262144


def test_load_model_dims_missing_path():
    md = load_model_dims("/no/such/dir", None)
    assert md.dims is None and md.max_model_len is None
    assert load_model_dims(None, 100).max_model_len == 100
