"""Optional non-conservative l=1 force head for the route-D baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DirectForceHead(nn.Module):
    """Read an equivariant Cartesian vector from e3nn l=1 channels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_weights = nn.Parameter(torch.empty(channels))
        nn.init.normal_(self.channel_weights, std=1.0 / max(1, channels) ** 0.5)

    def forward(self, l1_features: Tensor) -> Tensor:
        if l1_features.ndim != 3 or l1_features.shape[1] != 3:
            raise ValueError("l1_features must have shape [N,3,C]")
        return torch.einsum("nmc,c->nm", l1_features, self.channel_weights)

