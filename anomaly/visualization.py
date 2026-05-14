from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch


def normalize_map(amap: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    amap = amap.astype(np.float32)
    lo = float(np.percentile(amap, 1.0))
    hi = float(np.percentile(amap, 99.5))
    if hi <= lo:
        hi = float(amap.max())
        lo = float(amap.min())
    return np.clip((amap - lo) / (hi - lo + eps), 0.0, 1.0)


def tensor_to_gray_image(x: torch.Tensor) -> np.ndarray:
    """CxHxW tensor in [0,1] -> HxW uint8 grayscale."""
    x = x.detach().cpu().float()
    if x.ndim != 3:
        raise ValueError(f"Expected CxHxW tensor, got {tuple(x.shape)}")
    if x.shape[0] == 1:
        img = x[0]
    else:
        img = x.mean(dim=0)
    img = img.clamp(0, 1).numpy()
    return (img * 255).astype(np.uint8)


def colorize_heatmap(amap: np.ndarray) -> np.ndarray:
    """Return RGB uint8 heatmap using matplotlib's turbo/jet fallback."""
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("turbo")
        rgb = cmap(amap)[..., :3]
    except Exception:
        # Simple red heatmap fallback.
        rgb = np.zeros((*amap.shape, 3), dtype=np.float32)
        rgb[..., 0] = amap
        rgb[..., 1] = np.maximum(0, amap - 0.5) * 0.5
    return (rgb * 255).astype(np.uint8)


def save_anomaly_images(
    image_tensor: torch.Tensor,
    anomaly_map: torch.Tensor,
    stem: str,
    out_dir: Path,
    save_map: bool = True,
    save_overlay: bool = True,
    alpha: float = 0.45,
) -> Tuple[Optional[str], Optional[str]]:
    out_dir = Path(out_dir)
    maps_dir = out_dir / "maps"
    overlays_dir = out_dir / "overlays"
    if save_map:
        maps_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    gray = tensor_to_gray_image(image_tensor)
    amap = anomaly_map.detach().cpu().float().squeeze().numpy()
    amap = normalize_map(amap)
    heat = colorize_heatmap(amap)

    map_path = None
    overlay_path = None
    if save_map:
        map_path = str(maps_dir / f"{stem}_map.png")
        Image.fromarray(heat).save(map_path)
    if save_overlay:
        base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        overlay = (1.0 - alpha) * base + alpha * heat.astype(np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        overlay_path = str(overlays_dir / f"{stem}_overlay.png")
        Image.fromarray(overlay).save(overlay_path)
    return map_path, overlay_path
