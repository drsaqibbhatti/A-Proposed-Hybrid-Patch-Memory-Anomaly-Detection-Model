from typing import Dict, Optional

import torch
from tqdm import tqdm

from .utils import robust_quantile, sample_flat_values, topk_score


@torch.no_grad()
def calibrate_model(
    model,
    dataloader,
    device: torch.device,
    weights: Optional[Dict[str, float]] = None,
    topk_ratio: float = 0.01,
    threshold_quantile: float = 0.995,
    max_map_values_per_component: int = 100000,
    seed: int = 42,
    nn_chunk_size: int = 2048,
    memory_chunk_size: int = 20000,
) -> Dict:
    """Estimate normal-data statistics used to normalize maps and set a threshold."""
    model.eval()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    values: Dict[str, list] = {}
    scores: Dict[str, list] = {}

    for batch in tqdm(dataloader, desc="Calibrating component statistics"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        raw_maps = model.raw_anomaly_maps(
            images, nn_chunk_size=nn_chunk_size, memory_chunk_size=memory_chunk_size
        )
        for name, amap in raw_maps.items():
            scores.setdefault(name, []).append(topk_score(amap, topk_ratio=topk_ratio).detach().cpu())
            current_values = values.setdefault(name, [])
            already = sum(v.numel() for v in current_values)
            room = max(0, max_map_values_per_component - already)
            if room > 0:
                current_values.append(sample_flat_values(amap, room, generator=gen))

    components = {}
    present = sorted(scores.keys())
    for name in present:
        score_tensor = torch.cat(scores[name]).float()
        value_tensor = torch.cat(values.get(name, [torch.tensor([])])).float()
        if value_tensor.numel() == 0:
            value_tensor = score_tensor
        components[name] = {
            "map_mean": float(value_tensor.mean().item()),
            "map_std": float(value_tensor.std(unbiased=False).clamp_min(1e-8).item()),
            "map_q95": robust_quantile(value_tensor, 0.95),
            "map_q99": robust_quantile(value_tensor, 0.99),
            "map_q995": robust_quantile(value_tensor, 0.995),
            "score_mean": float(score_tensor.mean().item()),
            "score_std": float(score_tensor.std(unbiased=False).clamp_min(1e-8).item()),
            "score_q95": robust_quantile(score_tensor, 0.95),
            "score_q99": robust_quantile(score_tensor, 0.99),
            "score_q995": robust_quantile(score_tensor, 0.995),
        }

    if weights is None:
        weights = {name: 1.0 for name in present}
    weights = {name: float(weights.get(name, 0.0)) for name in present}
    if sum(weights.values()) <= 0:
        weights = {name: 1.0 for name in present}
    total = sum(weights.values())
    weights = {name: val / total for name, val in weights.items()}

    calibration = {
        "components": components,
        "weights": weights,
        "topk_ratio": float(topk_ratio),
        "threshold_quantile": float(threshold_quantile),
    }
    model.calibration = calibration

    combined_scores = []
    for batch in tqdm(dataloader, desc="Calibrating combined threshold"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        raw_maps = model.raw_anomaly_maps(
            images, nn_chunk_size=nn_chunk_size, memory_chunk_size=memory_chunk_size
        )
        combined = model.combine_maps(raw_maps, normalize=True)
        combined_scores.append(topk_score(combined, topk_ratio=topk_ratio).detach().cpu())

    combined_scores_t = torch.cat(combined_scores).float() if combined_scores else torch.tensor([0.0])
    calibration["combined"] = {
        "score_mean": float(combined_scores_t.mean().item()),
        "score_std": float(combined_scores_t.std(unbiased=False).clamp_min(1e-8).item()),
        "score_q95": robust_quantile(combined_scores_t, 0.95),
        "score_q99": robust_quantile(combined_scores_t, 0.99),
        "score_q995": robust_quantile(combined_scores_t, 0.995),
        "threshold": robust_quantile(combined_scores_t, threshold_quantile),
        "num_images": int(combined_scores_t.numel()),
    }
    model.calibration = calibration
    return calibration
