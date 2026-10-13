"""C3-envelope-normalized invariant attention."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from equiso2.irreps.layout import SO2EdgeTensor, SO2Layout


class SmoothEnvelopeAttention(nn.Module):
    r"""Apply the smooth null-state attention from ``whf.md``, Section 4.6.

    .. math::
        w_{ij,f}=\frac{s(r_{ij})^2e^{\ell_{ij,f}}}
        {\operatorname{softplus}(\zeta_f)+\sum_k s(r_{ik})^2e^{\ell_{ik,f}}}.

    The input/output edge tensor is ``[E,D,C]`` and returned diagnostic weights
    are ``[E,F]``.  No raw neighbor count or top-k normalization is used.
    """

    def __init__(
        self,
        layout: SO2Layout,
        channels: int,
        num_foci: int,
        radial_dim: int,
        logit_bound: float = 8.0,
    ) -> None:
        super().__init__()
        if channels % num_foci != 0:
            raise ValueError("channels must be divisible by num_foci")
        if logit_bound <= 0.0:
            raise ValueError("logit_bound must be positive")
        self.layout = layout
        self.channels = int(channels)
        self.num_foci = int(num_foci)
        self.focus_channels = channels // num_foci
        self.logit_bound = float(logit_bound)
        self.score_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.focus_channels + radial_dim, self.focus_channels),
                    nn.SiLU(),
                    nn.Linear(self.focus_channels, 1),
                )
                for _ in range(num_foci)
            ]
        )
        self.null_logits = nn.Parameter(torch.zeros(num_foci))

    def forward(
        self,
        edges: SO2EdgeTensor,
        radial: Tensor,
        envelope: Tensor,
        dst: Tensor,
        num_nodes: int,
    ) -> tuple[SO2EdgeTensor, Tensor]:
        if edges.layout != self.layout or edges.channels != self.channels:
            raise ValueError("edge tensor does not match attention configuration")
        if radial.ndim != 2 or radial.shape[0] != edges.num_edges:
            raise ValueError("radial must have shape [E, radial_dim]")
        if envelope.shape != (edges.num_edges,):
            raise ValueError("envelope must have shape [E]")
        if dst.shape != (edges.num_edges,):
            raise ValueError("dst must have shape [E]")
        scalar_index = self.layout.compact_index(0, 0)
        numerators: list[Tensor] = []
        for focus in range(self.num_foci):
            begin = focus * self.focus_channels
            end = begin + self.focus_channels
            scalar = edges.values[:, scalar_index, begin:end]
            score_input = torch.cat((scalar, radial), dim=1)
            raw = self.score_networks[focus](score_input).squeeze(1)
            bounded = self.logit_bound * torch.tanh(raw / self.logit_bound)
            numerator = envelope.square() * torch.exp(bounded)
            numerators.append(numerator)
        numerator_matrix = torch.stack(numerators, dim=1)
        denominator = numerator_matrix.new_zeros((num_nodes, self.num_foci))
        denominator = denominator.index_add(
            0, dst.to(device=numerator_matrix.device), numerator_matrix
        )
        null = functional.softplus(self.null_logits).reshape(1, self.num_foci)
        denominator = denominator + null.expand(num_nodes, self.num_foci)
        edge_denominator = denominator.index_select(
            0, dst.to(device=denominator.device)
        )
        weights = numerator_matrix / edge_denominator
        weighted_foci: list[Tensor] = []
        for focus in range(self.num_foci):
            begin = focus * self.focus_channels
            end = begin + self.focus_channels
            focus_values = edges.values[:, :, begin:end]
            focus_weight = weights[:, focus].reshape(edges.num_edges, 1, 1)
            weighted_foci.append(focus_values * focus_weight)
        return (
            SO2EdgeTensor(
                values=torch.cat(weighted_foci, dim=2),
                layout=edges.layout,
                topology_id=edges.topology_id,
                gauge_id=edges.gauge_id,
            ),
            weights,
        )

