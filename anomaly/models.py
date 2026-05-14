from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Optional, Tuple, cast

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import FrozenTeacher, default_layers_for_teacher
from .memory import nearest_neighbor_distance
from .utils import flatten_hw, topk_score


def _group_count(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(channels, channels, kernel_size=3, stride=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ConvPredictor(nn.Module):
    """Lightweight student that predicts frozen teacher patch embeddings from the image."""

    def __init__(self, out_channels: int, width: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(1, width, 3, 2),
            ResidualBlock(width),
            ConvGNAct(width, width * 2, 3, 2),
            ResidualBlock(width * 2),
            ConvGNAct(width * 2, width * 4, 3, 2),
            ResidualBlock(width * 4),
            ConvGNAct(width * 4, width * 4, 3, 1),
            ResidualBlock(width * 4),
            nn.Conv2d(width * 4, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        y = self.net(x)
        if y.shape[-2:] != target_size:
            y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
        return F.normalize(y, dim=1, eps=1e-6)


class FeatureAutoencoder(nn.Module):
    """Global feature autoencoder branch for logical/global anomalies."""

    def __init__(self, out_channels: int, width: int = 96):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvGNAct(1, width, 4, 2),
            ConvGNAct(width, width, 3, 1),
            ConvGNAct(width, width * 2, 4, 2),
            ConvGNAct(width * 2, width * 2, 3, 1),
            ConvGNAct(width * 2, width * 4, 4, 2),
            ResidualBlock(width * 4),
            ConvGNAct(width * 4, width * 4, 4, 2),
            ResidualBlock(width * 4),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(width * 4, width * 4, 3, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(width * 4, width * 2, 3, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(width * 2, width, 3, 1),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        z = self.encoder(x)
        y = self.decoder(z)
        if y.shape[-2:] != target_size:
            y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
        return F.normalize(y, dim=1, eps=1e-6)


class RandomProjector(nn.Module):
    """Fixed Gaussian projection for compact PatchCore memory and faster nearest-neighbor search."""

    def __init__(self, target_dim: int = 256, seed: int = 42):
        super().__init__()
        self.target_dim = int(target_dim)
        self.seed = int(seed)
        self.register_buffer("projection", torch.empty(0), persistent=True)

    def fit(self, in_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.target_dim <= 0 or self.target_dim >= in_dim:
            self.projection = torch.empty(0, device=device, dtype=dtype)
            return
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        mat = torch.randn(in_dim, self.target_dim, generator=gen, dtype=torch.float32)
        mat = mat / (self.target_dim ** 0.5)
        self.projection = mat.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.target_dim <= 0 or self.target_dim >= c:
            return x
        if self.projection.numel() == 0:
            self.fit(c, x.device, x.dtype)
        flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        y = flat @ self.projection.to(device=x.device, dtype=x.dtype)
        return y.reshape(b, h, w, self.target_dim).permute(0, 3, 1, 2).contiguous()


class PatchFeatureEmbedder(nn.Module):
    """Converts multi-layer teacher features into dense normalized patch embeddings."""

    def __init__(
        self,
        teacher_name: str = "wide_resnet50_2",
        pretrained: bool = True,
        teacher_ckpt: Optional[str] = None,
        layers: Optional[Iterable[str]] = None,
        target_dim: int = 256,
        local_agg_kernel: int = 3,
        projector_seed: int = 42,
        pdn_out_channels: int = 384,
        pdn_padding: bool = True,
    ):
        super().__init__()
        self.teacher_name = teacher_name
        self.layers = list(layers) if layers is not None else default_layers_for_teacher(teacher_name)
        self.teacher = FrozenTeacher(
            name=teacher_name,
            pretrained=pretrained,
            layers=self.layers,
            teacher_ckpt=teacher_ckpt,
            pdn_out_channels=pdn_out_channels,
            pdn_padding=pdn_padding,
        )
        self.local_agg_kernel = int(local_agg_kernel)
        self.projector = RandomProjector(target_dim=target_dim, seed=projector_seed)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.teacher(x)
        maps = []
        base_size = None
        for name in self.layers:
            if name not in features:
                raise KeyError(f"Teacher did not return layer '{name}'. Available: {list(features.keys())}")
            f = features[name]
            if base_size is None:
                base_size = f.shape[-2:]
            elif f.shape[-2:] != base_size:
                f = F.interpolate(f, size=base_size, mode="bilinear", align_corners=False)
            maps.append(f)
        embedding = torch.cat(maps, dim=1)
        if self.local_agg_kernel > 1:
            pad = self.local_agg_kernel // 2
            embedding = F.avg_pool2d(embedding, kernel_size=self.local_agg_kernel, stride=1, padding=pad)
        embedding = self.projector(embedding)
        return F.normalize(embedding, dim=1, eps=1e-6)


class AegisAD(nn.Module):
    """
    Hybrid anomaly detector:
      1. PatchCore-style memory bank over frozen teacher patch features.
      2. Student-teacher feature prediction branch.
      3. Feature autoencoder branch.

    The final anomaly map is a calibrated weighted combination of available component maps.
    """

    def __init__(
        self,
        teacher_name: str = "wide_resnet50_2",
        pretrained: bool = True,
        teacher_ckpt: Optional[str] = None,
        layers: Optional[Iterable[str]] = None,
        target_dim: int = 256,
        local_agg_kernel: int = 3,
        enable_student: bool = True,
        enable_autoencoder: bool = True,
        student_width: int = 96,
        ae_width: int = 96,
        image_size: int = 256,
        projector_seed: int = 42,
        pdn_out_channels: int = 384,
        pdn_padding: bool = True,
    ):
        super().__init__()
        self.model_config = dict(
            teacher_name=teacher_name,
            pretrained=pretrained,
            teacher_ckpt=teacher_ckpt,
            layers=list(layers) if layers is not None else default_layers_for_teacher(teacher_name),
            target_dim=target_dim,
            local_agg_kernel=local_agg_kernel,
            enable_student=enable_student,
            enable_autoencoder=enable_autoencoder,
            student_width=student_width,
            ae_width=ae_width,
            image_size=image_size,
            projector_seed=projector_seed,
            pdn_out_channels=pdn_out_channels,
            pdn_padding=pdn_padding,
        )
        self.embedder = PatchFeatureEmbedder(
            teacher_name=teacher_name,
            pretrained=pretrained,
            teacher_ckpt=teacher_ckpt,
            layers=cast(List[str], self.model_config["layers"]),
            target_dim=target_dim,
            local_agg_kernel=local_agg_kernel,
            projector_seed=projector_seed,
            pdn_out_channels=pdn_out_channels,
            pdn_padding=pdn_padding,
        )
        self.enable_student = bool(enable_student)
        self.enable_autoencoder = bool(enable_autoencoder)
        self.student = ConvPredictor(target_dim, width=student_width) if self.enable_student else None
        self.autoencoder = FeatureAutoencoder(target_dim, width=ae_width) if self.enable_autoencoder else None
        self.memory_bank: Optional[torch.Tensor] = None
        self.calibration: Dict = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.embedder.eval()
        return self

    def branch_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x
        if x.shape[1] == 3:
            return x.mean(dim=1, keepdim=True)
        raise ValueError(f"Expected 1 or 3 input channels, got {x.shape[1]}")

    @torch.no_grad()
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder(x)

    def training_predictions(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            target = self.extract_embedding(x)
        target_size = target.shape[-2:]
        inp = self.branch_input(x)
        student_pred = self.student(inp, target_size) if self.student is not None else None
        ae_pred = self.autoencoder(inp, target_size) if self.autoencoder is not None else None
        return target, student_pred, ae_pred

    def set_memory(self, memory_bank: Optional[torch.Tensor]) -> None:
        self.memory_bank = None if memory_bank is None else memory_bank.detach().float().cpu()

    def move_memory(self, device: torch.device) -> None:
        if self.memory_bank is not None:
            self.memory_bank = self.memory_bank.to(device)

    def _patchcore_map_from_embedding(
        self,
        embedding: torch.Tensor,
        image_size: Tuple[int, int],
        nn_chunk_size: int = 2048,
        memory_chunk_size: int = 20000,
    ) -> torch.Tensor:
        if self.memory_bank is None:
            raise RuntimeError("PatchCore memory bank is empty. Train/build memory before inference.")
        b, _, h, w = embedding.shape
        patches = flatten_hw(embedding)
        memory = self.memory_bank.to(device=embedding.device, dtype=embedding.dtype)
        distances = nearest_neighbor_distance(
            patches,
            memory,
            chunk_size=nn_chunk_size,
            memory_chunk_size=memory_chunk_size,
        )
        amap = distances.reshape(b, h, w).unsqueeze(1)
        return F.interpolate(amap, size=image_size, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def raw_anomaly_maps(
        self,
        x: torch.Tensor,
        nn_chunk_size: int = 2048,
        memory_chunk_size: int = 20000,
    ) -> Dict[str, torch.Tensor]:
        image_size: Tuple[int, int] = (x.shape[-2], x.shape[-1])
        target = self.extract_embedding(x)
        target_size = target.shape[-2:]
        maps: Dict[str, torch.Tensor] = {}

        if self.memory_bank is not None:
            maps["patchcore"] = self._patchcore_map_from_embedding(
                target, image_size=image_size, nn_chunk_size=nn_chunk_size, memory_chunk_size=memory_chunk_size
            )

        inp = self.branch_input(x)
        if self.student is not None:
            student_pred = self.student(inp, target_size)
            st_map = (student_pred - target).pow(2).mean(dim=1, keepdim=True)
            maps["student"] = F.interpolate(st_map, size=image_size, mode="bilinear", align_corners=False)

        if self.autoencoder is not None:
            ae_pred = self.autoencoder(inp, target_size)
            ae_map = (ae_pred - target).pow(2).mean(dim=1, keepdim=True)
            maps["autoencoder"] = F.interpolate(ae_map, size=image_size, mode="bilinear", align_corners=False)

        return maps

    def normalize_component_map(self, name: str, amap: torch.Tensor) -> torch.Tensor:
        component_stats = self.calibration.get("components", {}).get(name)
        if not component_stats:
            return amap
        mean = float(component_stats.get("map_mean", 0.0))
        std = float(component_stats.get("map_std", 1.0))
        z = (amap - mean) / (std + 1e-8)
        return torch.clamp(z, min=0.0)

    def combine_maps(self, maps: Dict[str, torch.Tensor], normalize: bool = True) -> torch.Tensor:
        if not maps:
            raise RuntimeError("No anomaly maps were produced. Enable PatchCore memory, student, or autoencoder.")
        weights = self.calibration.get("weights", {})
        if not weights:
            weights = {name: 1.0 / len(maps) for name in maps}
        combined = None
        total_weight = 0.0
        for name, amap in maps.items():
            w = float(weights.get(name, 0.0))
            if w <= 0:
                continue
            nm = self.normalize_component_map(name, amap) if normalize else amap
            combined = nm * w if combined is None else combined + nm * w
            total_weight += w
        if combined is None:
            first = next(iter(maps.values()))
            combined = torch.zeros_like(first)
        if total_weight > 0:
            combined = combined / total_weight
        return combined

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        topk_ratio: Optional[float] = None,
        normalize: bool = True,
        nn_chunk_size: int = 2048,
        memory_chunk_size: int = 20000,
    ) -> Dict[str, torch.Tensor]:
        topk_ratio = float(topk_ratio if topk_ratio is not None else self.calibration.get("topk_ratio", 0.01))
        raw_maps = self.raw_anomaly_maps(
            x, nn_chunk_size=nn_chunk_size, memory_chunk_size=memory_chunk_size
        )
        component_maps = {
            name: (self.normalize_component_map(name, amap) if normalize else amap)
            for name, amap in raw_maps.items()
        }
        combined = self.combine_maps(raw_maps, normalize=normalize)
        out: Dict[str, torch.Tensor] = {"anomaly_map": combined, "score": topk_score(combined, topk_ratio=topk_ratio)}
        for name, amap in raw_maps.items():
            out[f"{name}_map"] = amap
            out[f"{name}_score"] = topk_score(amap, topk_ratio=topk_ratio)
        for name, amap in component_maps.items():
            out[f"{name}_map_norm"] = amap
        return out

    def prepare_dynamic_buffers_for_load(self, state_dict: Dict[str, torch.Tensor]) -> None:
        key = "embedder.projector.projection"
        if key in state_dict and state_dict[key].numel() > 0:
            self.embedder.projector.projection = state_dict[key].detach().clone()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, map_location: str | torch.device = "cpu") -> Tuple["AegisAD", Dict]:
        payload = torch.load(checkpoint_path, map_location=map_location)
        cfg = copy.deepcopy(payload["config"]["model"])
        # The checkpoint already contains teacher weights; avoid re-downloading ImageNet weights on load.
        cfg["pretrained"] = False
        cfg["teacher_ckpt"] = None
        model = cls(**cfg)
        state = payload["model_state"]
        model.prepare_dynamic_buffers_for_load(state)
        model.load_state_dict(state, strict=False)
        model.set_memory(payload.get("memory_bank"))
        model.calibration = payload.get("calibration", {})
        return model, payload
