import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def save_json(obj: Dict[str, Any], path: os.PathLike) -> None:
    def convert(o: Any) -> Any:
        if isinstance(o, torch.Tensor):
            if o.numel() == 1:
                return float(o.detach().cpu().item())
            return o.detach().cpu().tolist()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o)!r} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)


def load_json(path: os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "teacher", "net"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Checkpoint does not contain a valid state_dict-like object")


def _filter_compatible_state_dict(module: torch.nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    module_state = module.state_dict()
    compatible = {}
    for key, value in state.items():
        if key in module_state and tuple(module_state[key].shape) == tuple(value.shape):
            compatible[key] = value
    return compatible


def load_partial_state_dict(module: torch.nn.Module, checkpoint_path: str, strict: bool = False) -> Tuple[list, list]:
    """Load a checkpoint robustly, tolerating common wrappers/prefixes.

    This is intentionally conservative: it chooses the prefix-cleaning candidate with the
    largest number of shape-compatible keys, so PDN checkpoints saved as either a bare
    Sequential (`0.weight`) or wrapped module (`net.0.weight`) can both load.
    """
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = extract_state_dict(payload)
    candidates = [
        state,
        strip_prefix_if_present(state, prefixes=("module.", "model.", "teacher.", "backbone.")),
        strip_prefix_if_present(state, prefixes=("module.", "model.", "teacher.", "backbone.", "net.")),
        strip_prefix_if_present(state, prefixes=("pdn.",)),
        strip_prefix_if_present(state, prefixes=("module.", "model.", "teacher.", "backbone.", "pdn.")),
    ]
    best = max(candidates, key=lambda sd: len(_filter_compatible_state_dict(module, sd)))
    compatible = _filter_compatible_state_dict(module, best)
    if strict and len(compatible) != len(module.state_dict()):
        missing = sorted(set(module.state_dict().keys()) - set(compatible.keys()))
        raise RuntimeError(f"Strict checkpoint loading failed; missing compatible keys: {missing[:10]}")
    missing, unexpected = module.load_state_dict(compatible, strict=False)
    return list(missing), list(unexpected)


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def flatten_hw(x: torch.Tensor) -> torch.Tensor:
    """B,C,H,W -> B*H*W,C."""
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


def unflatten_hw(x: torch.Tensor, batch: int, height: int, width: int) -> torch.Tensor:
    """B*H*W,C -> B,C,H,W."""
    channels = x.shape[-1]
    return x.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()


def topk_score(anomaly_map: torch.Tensor, topk_ratio: float = 0.01) -> torch.Tensor:
    """Image-level anomaly score from a Bx1xHxW map using mean of top-k pixels."""
    if anomaly_map.ndim != 4:
        raise ValueError(f"Expected Bx1xHxW anomaly map, got shape {tuple(anomaly_map.shape)}")
    flat = anomaly_map.flatten(1)
    k = max(1, int(math.ceil(flat.shape[1] * topk_ratio)))
    return flat.topk(k, dim=1).values.mean(dim=1)


def robust_quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    q = min(max(float(q), 0.0), 1.0)
    return float(torch.quantile(values.float().cpu(), q).item())


def sample_flat_values(x: torch.Tensor, max_values: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    flat = x.detach().flatten().float().cpu()
    if flat.numel() <= max_values:
        return flat
    idx = torch.randperm(flat.numel(), generator=generator)[:max_values]
    return flat[idx]
