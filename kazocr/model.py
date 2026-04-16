from __future__ import annotations

import torch
from torch import nn

from .config import OCRConfig


class CRNN(nn.Module):
    def __init__(self, vocab_size: int, config: OCRConfig) -> None:
        super().__init__()
        self.config = config
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, config.cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.cnn_channels),
            nn.ReLU(inplace=True),
        )
        self.sequence_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(
            input_size=config.cnn_channels,
            hidden_size=config.rnn_hidden,
            num_layers=config.rnn_layers,
            dropout=config.dropout if config.rnn_layers > 1 else 0.0,
            batch_first=False,
            bidirectional=True,
        )
        self.classifier = nn.Linear(config.rnn_hidden * 2, vocab_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = self.sequence_pool(features).squeeze(2)
        features = features.permute(2, 0, 1)
        sequence, _ = self.rnn(features)
        logits = self.classifier(sequence)
        return logits
