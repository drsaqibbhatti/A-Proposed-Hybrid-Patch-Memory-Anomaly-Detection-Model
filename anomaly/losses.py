from typing import Dict

import torch
import torch.nn.functional as F


def hard_feature_loss(pred: torch.Tensor, target: torch.Tensor, hard_ratio: float = 0.10) -> torch.Tensor:
    """
    EfficientAD-style hard feature loss.

    Computes per-patch squared feature error and averages only the hardest patches. This prevents
    the student from wasting most gradient on already-easy normal regions.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    error = (pred - target.detach()).pow(2).mean(dim=1)  # B,H,W
    flat = error.flatten()
    k = max(1, int(flat.numel() * hard_ratio))
    return flat.topk(k).values.mean()


def feature_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    return F.mse_loss(pred, target.detach())


def consistency_loss(student: torch.Tensor, autoencoder: torch.Tensor) -> torch.Tensor:
    if student.shape != autoencoder.shape:
        raise ValueError(f"Shape mismatch: student={tuple(student.shape)} autoencoder={tuple(autoencoder.shape)}")
    return F.mse_loss(student, autoencoder.detach())


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> torch.Tensor:
    """Small dependency-free SSIM loss for optional reconstruction experiments."""
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x={tuple(x.shape)} y={tuple(y.shape)}")
    channels = x.shape[1]
    pad = window_size // 2
    weight = torch.ones(channels, 1, window_size, window_size, device=x.device, dtype=x.dtype)
    weight = weight / (window_size * window_size)

    mu_x = F.conv2d(x, weight, padding=pad, groups=channels)
    mu_y = F.conv2d(y, weight, padding=pad, groups=channels)
    sigma_x = F.conv2d(x * x, weight, padding=pad, groups=channels) - mu_x.pow(2)
    sigma_y = F.conv2d(y * y, weight, padding=pad, groups=channels) - mu_y.pow(2)
    sigma_xy = F.conv2d(x * y, weight, padding=pad, groups=channels) - mu_x * mu_y

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return 1.0 - ssim.mean()


def weighted_training_loss(
    student_pred: torch.Tensor,
    ae_pred: torch.Tensor,
    target: torch.Tensor,
    hard_ratio: float = 0.10,
    student_weight: float = 1.0,
    autoencoder_weight: float = 0.5,
    consistency_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=target.device)

    if student_pred is not None:
        losses["student_hard"] = hard_feature_loss(student_pred, target, hard_ratio=hard_ratio)
        total = total + student_weight * losses["student_hard"]

    if ae_pred is not None:
        losses["autoencoder_mse"] = feature_mse_loss(ae_pred, target)
        total = total + autoencoder_weight * losses["autoencoder_mse"]

    if student_pred is not None and ae_pred is not None and consistency_weight > 0:
        losses["student_ae_consistency"] = consistency_loss(student_pred, ae_pred)
        total = total + consistency_weight * losses["student_ae_consistency"]

    losses["total"] = total
    return losses
