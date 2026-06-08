"""Lightweight 1D CNN for beat-level disease classification."""

import torch
import torch.nn as nn


class BeatCNNClassifier(nn.Module):
    """
    Simple 1D CNN — no transformers, LSTMs, or attention.

    Input:  (batch, 1, 70)
    Output: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 17, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block(x))
