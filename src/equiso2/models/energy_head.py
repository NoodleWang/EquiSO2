"""Invariant per-atom scalar energy head."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AtomicEnergyHead(nn.Module):
    """Map ``l=0`` channels ``[N,C]`` to per-atom and graph energies."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, max(1, channels // 2)),
            nn.SiLU(),
            nn.Linear(max(1, channels // 2), 1),
        )

    def forward(
        self, scalar_features: Tensor, batch: Tensor, num_graphs: int
    ) -> tuple[Tensor, Tensor]:
        if scalar_features.ndim != 2:
            raise ValueError("scalar_features must have shape [N,C]")
        per_atom = self.network(scalar_features).squeeze(1)
        energy = per_atom.new_zeros((num_graphs,))
        energy = energy.index_add(0, batch.to(device=per_atom.device), per_atom)
        return energy, per_atom

