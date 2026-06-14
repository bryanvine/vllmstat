from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class ModelDims:
    dims: dict[str, int] | None
    max_model_len: int | None


def dims_from_config(config: dict) -> dict[str, int] | None:
    """Extract KV dims (layers, kv_heads, head_dim) from a HF model config dict."""
    try:
        layers = int(config["num_hidden_layers"])
        n_attn = int(config["num_attention_heads"])
        kv_heads = int(config.get("num_key_value_heads") or n_attn)
        head_dim = config.get("head_dim")
        head_dim = int(head_dim) if head_dim else int(config["hidden_size"]) // n_attn
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if layers <= 0 or kv_heads <= 0 or head_dim <= 0:
        return None
    return {"layers": layers, "kv_heads": kv_heads, "head_dim": head_dim}


def load_model_dims(root: str | None, max_model_len: int | None) -> ModelDims:
    """Read <root>/config.json for KV dims when the model path is locally readable."""
    dims = None
    if root and os.path.isdir(root):
        cfg = os.path.join(root, "config.json")
        if os.path.isfile(cfg):
            try:
                with open(cfg) as f:
                    dims = dims_from_config(json.load(f))
            except (OSError, json.JSONDecodeError):
                dims = None
    return ModelDims(dims=dims, max_model_len=max_model_len)
