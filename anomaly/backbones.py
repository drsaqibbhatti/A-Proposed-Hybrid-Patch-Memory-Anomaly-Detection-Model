from collections import OrderedDict
from typing import Dict, Iterable, List, Literal, Optional, cast

import torch
from torch import nn

from .pdnS import build_pdn
from .utils import load_partial_state_dict


_TORCHVISION_DEFAULT_LAYERS = {
    "resnet18": ["layer2", "layer3"],
    "resnet34": ["layer2", "layer3"],
    "resnet50": ["layer2", "layer3"],
    "wide_resnet50_2": ["layer2", "layer3"],
    "efficientnet_b0": ["features.4", "features.6"],
}


def default_layers_for_teacher(name: str) -> List[str]:
    if name in ("pdn_s", "pdn_m"):
        return ["pdn"]
    return list(_TORCHVISION_DEFAULT_LAYERS.get(name, ["layer2", "layer3"]))


def _torchvision_imports():
    try:
        from torchvision import models
        from torchvision.models.feature_extraction import create_feature_extractor
        return models, create_feature_extractor
    except Exception as exc:
        raise RuntimeError(
            "Torchvision could not be imported. Install a torch/torchvision pair that matches your CUDA/Python "
            "environment, or use --teacher pdn_s/--teacher pdn_m with --teacher-ckpt to avoid torchvision. "
            f"Original error: {exc}"
        ) from exc


def _build_torchvision_model(name: str, pretrained: bool) -> nn.Module:
    models, _ = _torchvision_imports()
    weights = None
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet18(weights=weights)
    if name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet34(weights=weights)
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        return models.resnet50(weights=weights)
    if name == "wide_resnet50_2":
        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
        return models.wide_resnet50_2(weights=weights)
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        return models.efficientnet_b0(weights=weights)
    raise ValueError(
        f"Unsupported teacher '{name}'. Use one of: resnet18, resnet34, resnet50, "
        "wide_resnet50_2, efficientnet_b0, pdn_s, pdn_m."
    )


class FrozenTeacher(nn.Module):
    """Frozen feature teacher that accepts 1-channel or 3-channel tensors."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        name: str = "wide_resnet50_2",
        pretrained: bool = True,
        layers: Optional[Iterable[str]] = None,
        teacher_ckpt: Optional[str] = None,
        pdn_out_channels: int = 384,
        pdn_padding: bool = True,
    ):
        super().__init__()
        self.name = name
        self.layers = list(layers) if layers is not None else default_layers_for_teacher(name)
        self.is_pdn = name in ("pdn_s", "pdn_m")

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if self.is_pdn:
            self.net = build_pdn(cast(Literal["pdn_s", "pdn_m"], name), out_channels=pdn_out_channels, padding=pdn_padding)
            if teacher_ckpt:
                missing, unexpected = load_partial_state_dict(self.net, teacher_ckpt, strict=False)
                print(f"Loaded PDN teacher checkpoint. Missing={len(missing)}, unexpected={len(unexpected)}")
            self.extractor = None
        else:
            _, create_feature_extractor = _torchvision_imports()
            self.net = _build_torchvision_model(name, pretrained=pretrained)
            if teacher_ckpt:
                missing, unexpected = load_partial_state_dict(self.net, teacher_ckpt, strict=False)
                print(f"Loaded teacher checkpoint. Missing={len(missing)}, unexpected={len(unexpected)}")
            return_nodes = {layer: layer for layer in self.layers}
            self.extractor = create_feature_extractor(self.net, return_nodes=return_nodes)

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BxCxHxW image tensor, got {tuple(x.shape)}")
        if self.is_pdn:
            # PDN teacher was pre-trained on grayscale (1-channel) input.
            # Convert RGB -> grayscale if needed, but do not expand to 3 channels.
            if x.shape[1] == 3:
                x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
            elif x.shape[1] != 1:
                raise ValueError(f"Expected 1 or 3 input channels, got {x.shape[1]}")
            return x
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 input channels, got {x.shape[1]}")
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.preprocess(x)
        if self.is_pdn:
            return OrderedDict({"pdn": self.net(x)})
        assert self.extractor is not None
        features = self.extractor(x)
        return OrderedDict((key, features[key]) for key in self.layers)
