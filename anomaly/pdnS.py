from typing import Literal

import torch
import torch.nn.functional as F


class PdnS(torch.nn.Module):

    def __init__(self, last_kernel_size=384, with_bn=False) -> None:
        super().__init__()
        # Layer Name Stride Kernel Size Number of Kernels Padding Activation
        # Conv-1 1×1 4×4 128 3 ReLU
        # AvgPool-1 2×2 2×2 128 1 -
        # Conv-2 1×1 4×4 256 3 ReLU
        # AvgPool-2 2×2 2×2 256 1 -
        # Conv-3 1×1 3×3 256 1 ReLU
        # Conv-4 1×1 4×4 384 0 -
        self.with_bn = with_bn
        self.conv1 = torch.nn.Conv2d(1, 128, kernel_size=4, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(256, last_kernel_size, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
        if self.with_bn:
            self.bn1 = torch.nn.BatchNorm2d(128)
            self.bn2 = torch.nn.BatchNorm2d(256)
            self.bn3 = torch.nn.BatchNorm2d(256)
            self.bn4 = torch.nn.BatchNorm2d(last_kernel_size)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.bn3(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x) if self.with_bn else x
        return x


def build_pdn(name: Literal["pdn_s", "pdn_m"], out_channels: int = 384, padding: bool = True) -> torch.nn.Module:
    """Compatibility wrapper so backbones.py can call build_pdn from either pdn.py or pdnS.py.

    Both pdn_s and pdn_m map to PdnS here. The `padding` flag maps to `with_bn`
    in PdnS — pass --pdn-padding when your checkpoint was trained with BatchNorm enabled.
    """
    if name not in ("pdn_s", "pdn_m"):
        raise ValueError(f"Unknown PDN name: {name}")
    return PdnS(last_kernel_size=out_channels, with_bn=padding)

