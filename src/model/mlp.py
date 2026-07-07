"""
MLP regressor: 256 → 100 → 1 (sigmoid output, MSE loss, Adam).
Matches the architecture of Lézoray & Nouri MMSP 2025.
"""
import torch
import torch.nn as nn


class SaliencyMLP(nn.Module):
    def __init__(self, in_dim: int = 9, hidden: list = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 100]
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
