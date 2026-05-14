import os
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class AnoDataset(Dataset):
    """
    Drop-in version of your dataloader.

    It recursively loads images from a root directory, converts them to grayscale, and returns
    either image or (image, filename). No labels are stored here because training is normal-only.
    """

    def __init__(self, path: str = "", transform: Optional[Callable] = None, return_filename: bool = False):
        self.path = path
        self.transform = transform
        self.return_filename = return_filename

        exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
        self.image_paths: List[str] = []
        for root, _, files in os.walk(self.path):
            for file_name in sorted(files):
                if file_name.lower().endswith(exts):
                    self.image_paths.append(os.path.join(root, file_name))
        self.total_images = len(self.image_paths)

    def __len__(self) -> int:
        return self.total_images

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        if self.return_filename:
            return image, os.path.basename(image_path)
        return image


class LabeledAnoDataset(Dataset):
    """
    Evaluation dataset that can combine normal and defect roots.

    Labels:
        normal -> 0
        defect -> 1
        unlabeled -> -1
    """

    def __init__(
        self,
        normal_dir: Optional[str] = None,
        defect_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        if data_dir:
            self.samples.extend((p, -1) for p in list_images(data_dir))
        if normal_dir:
            self.samples.extend((p, 0) for p in list_images(normal_dir))
        if defect_dir:
            self.samples.extend((p, 1) for p in list_images(defect_dir))
        self.samples.sort(key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label, os.path.basename(image_path), image_path


def list_images(path: Union[str, os.PathLike]) -> List[str]:
    exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
    result: List[str] = []
    path = str(path)
    for root, _, files in os.walk(path):
        for file_name in sorted(files):
            if file_name.lower().endswith(exts):
                result.append(os.path.join(root, file_name))
    return result


class ResizeToTensor:
    """Minimal torchvision-free transform: PIL grayscale image -> 1xHxW float tensor."""

    def __init__(self, size: Union[int, Sequence[int]] = 256):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            size = tuple(size)
            if len(size) != 2:
                raise ValueError("image_size must be an int or a two-value sequence")
            self.size = size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.size[1], self.size[0]), resample=Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).contiguous()


def build_transform(image_size: Union[int, Sequence[int]] = 256) -> Callable:
    return ResizeToTensor(image_size)


def collate_eval(batch):
    images, labels, names, paths = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long), list(names), list(paths)
