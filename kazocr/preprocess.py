from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def resize_preserving_aspect(image: Image.Image, target_height: int, min_width: int, max_width: int) -> Image.Image:
    scale = target_height / max(1, image.height)
    width = int(image.width * scale)
    width = max(min_width, min(max_width, width))
    return image.resize((width, target_height), Image.BICUBIC)


def load_image(path: str | Path) -> Image.Image:
    image = Image.open(path).convert("L")
    image = ImageOps.autocontrast(image)
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array).unsqueeze(0)
