from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OCRConfig:
    image_height: int = 48
    min_width: int = 96
    max_width: int = 384
    cnn_channels: int = 256
    rnn_hidden: int = 256
    rnn_layers: int = 2
    dropout: float = 0.1
