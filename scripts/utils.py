import os, json, random
import torch, numpy as np
from copy import deepcopy
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def deep_update(base, updates):
    out = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and k in out:
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
        
    return out


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def env_flag(name, default=False):
    v = os.getenv(name)
    if v is None: return default
    return v.lower() in ("1", "true", "yes", "y")