from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from torch.utils.data import Dataset

from .charset import KazakhLatinCharset
from .config import OCRConfig
from .preprocess import image_to_tensor, load_image, resize_preserving_aspect
from .text import load_corpus, sample_text


def _discover_fonts(fonts_dir: str | None) -> list[str]:
    candidates: list[str] = []
    if fonts_dir:
        candidates.extend(str(path) for path in Path(fonts_dir).rglob("*.ttf"))
        candidates.extend(str(path) for path in Path(fonts_dir).rglob("*.otf"))
    windows_fonts = Path("C:/Windows/Fonts")
    if windows_fonts.exists():
        for name in ["arial.ttf", "times.ttf", "calibri.ttf", "cambria.ttc", "verdana.ttf", "segoeui.ttf"]:
            path = windows_fonts / name
            if path.exists():
                candidates.append(str(path))
    return list(dict.fromkeys(candidates))


def _random_font(font_paths: list[str], size: int) -> ImageFont.ImageFont:
    if font_paths:
        try:
            return ImageFont.truetype(random.choice(font_paths), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _render_text_line(text: str, config: OCRConfig, font_paths: list[str]) -> Image.Image:
    canvas_width = config.max_width
    canvas_height = config.image_height
    background = random.randint(235, 255)
    foreground = random.randint(0, 35)
    image = Image.new("L", (canvas_width, canvas_height), color=background)
    draw = ImageDraw.Draw(image)
    font = _random_font(font_paths, size=random.randint(24, 34))
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x_max = max(2, canvas_width - text_width - 2)
    y_max = max(2, canvas_height - text_height - 2)
    x = random.randint(2, x_max)
    y = random.randint(1, y_max)
    draw.text((x, y), text, fill=foreground, font=font)

    if random.random() < 0.3:
        image = image.rotate(random.uniform(-1.2, 1.2), expand=False, fillcolor=background)
    if random.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.9)))
    if random.random() < 0.2:
        image = ImageOps.autocontrast(image)
    return image


class SyntheticKazakhDataset(Dataset):
    def __init__(
        self,
        size: int,
        charset: KazakhLatinCharset,
        config: OCRConfig,
        corpus_path: str | None = None,
        fonts_dir: str | None = None,
    ) -> None:
        self.size = size
        self.charset = charset
        self.config = config
        self.corpus = load_corpus(corpus_path)
        self.font_paths = _discover_fonts(fonts_dir)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        text = sample_text(self.corpus)
        image = _render_text_line(text, self.config, self.font_paths)
        image = resize_preserving_aspect(image, self.config.image_height, self.config.min_width, self.config.max_width)
        tensor = image_to_tensor(image)
        encoded = torch.tensor(self.charset.encode(text), dtype=torch.long)
        return {"image": tensor, "text": text, "target": encoded, "width": tensor.shape[-1]}


class ManifestOCRDataset(Dataset):
    def __init__(self, manifest_path: str, charset: KazakhLatinCharset, config: OCRConfig) -> None:
        self.charset = charset
        self.config = config
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.samples: list[tuple[Path, str]] = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_rel, text = line.split("\t", 1)
            self.samples.append((self.root / image_rel, text.strip()))
        if not self.samples:
            raise ValueError(f"No samples found in manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        image_path, text = self.samples[index]
        image = load_image(image_path)
        image = resize_preserving_aspect(image, self.config.image_height, self.config.min_width, self.config.max_width)
        tensor = image_to_tensor(image)
        encoded = torch.tensor(self.charset.encode(text), dtype=torch.long)
        return {"image": tensor, "text": text, "target": encoded, "width": tensor.shape[-1]}


def collate_ocr_batch(items: Iterable[dict[str, torch.Tensor | str | int]]) -> dict[str, torch.Tensor | list[str]]:
    batch = list(items)
    widths = [int(item["width"]) for item in batch]
    max_width = max(widths)
    images = []
    targets = []
    target_lengths = []
    texts = []
    for item in batch:
        image = item["image"]
        pad = max_width - image.shape[-1]
        if pad > 0:
            image = torch.nn.functional.pad(image, (0, pad), value=1.0)
        images.append(image)
        target = item["target"]
        targets.append(target)
        target_lengths.append(target.shape[0])
        texts.append(str(item["text"]))
    images_tensor = torch.stack(images)
    targets_tensor = torch.cat(targets)
    input_lengths = torch.full((len(batch),), fill_value=max(1, math.ceil(max_width / 4)), dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)
    return {
        "images": images_tensor,
        "targets": targets_tensor,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths_tensor,
        "texts": texts,
    }
