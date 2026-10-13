"""DPA4-style edge-conditioned low-rank SO(2) degree product."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from equiso2.irreps.layout import SO2EdgeTensor, SO2Layout


class LowRankSO2DegreeMixer(nn.Module):
    r"""Channel-diagonal, edge-conditioned cross-degree mixer.

    This is the named implementation of ``whf.md``, Section 4.3:

    .. math::
        y_{lmc}=\sum_{l',r}K^{(r)}_{ll'|m|}(\rho)B_{rc}x_{l'mc}.

    Input/output values have shape ``[E,D_{L,M},C]``; radial features are
    ``[E,d_rho]`` and the envelope is ``[E]``.  Positive and negative members of
    each ``|m|>0`` pair share exactly the same kernel and channel factors, which
    makes the operator commute with local-y SO(2) rotations in the e3nn basis.
    """

    def __init__(
        self,
        layout: SO2Layout,
        channels: int,
        radial_dim: int,
        rank: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if channels <= 0 or radial_dim <= 0 or rank <= 0 or hidden_dim <= 0:
            raise ValueError("all dimensions and rank must be positive")
        self.layout = layout
        self.channels = int(channels)
        self.rank = int(rank)
        self.kernel_heads = nn.ModuleDict()
        self.channel_factors = nn.ParameterDict()
        self.residual_scales = nn.ParameterDict()
        for abs_order in range(layout.mmax + 1):
            degree_count = layout.lmax - abs_order + 1
            self.kernel_heads[str(abs_order)] = nn.Sequential(
                nn.Linear(radial_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, rank * degree_count * degree_count),
            )
            factor = nn.Parameter(torch.empty(rank, channels))
            nn.init.normal_(factor, mean=0.0, std=1.0 / math.sqrt(rank))
            self.channel_factors[str(abs_order)] = factor
            self.residual_scales[str(abs_order)] = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, edges: SO2EdgeTensor, radial: Tensor, envelope: Tensor
    ) -> SO2EdgeTensor:
        if edges.layout != self.layout:
            raise ValueError("edge tensor layout does not match mixer layout")
        if edges.channels != self.channels:
            raise ValueError("edge channel count does not match mixer")
        if radial.ndim != 2 or radial.shape[0] != edges.num_edges:
            raise ValueError("radial must have shape [E, radial_dim]")
        if envelope.shape != (edges.num_edges,):
            raise ValueError("envelope must have shape [E]")

        output = edges.values
        edge_count = edges.num_edges
        for abs_order in range(self.layout.mmax + 1):
            degree_count = self.layout.lmax - abs_order + 1
            kernel = self.kernel_heads[str(abs_order)](radial)
            kernel = torch.tanh(
                kernel.reshape(
                    edge_count, self.rank, degree_count, degree_count
                )
            )
            kernel = kernel * envelope.reshape(edge_count, 1, 1, 1)
            factors = self.channel_factors[str(abs_order)]
            scale = self.residual_scales[str(abs_order)] / math.sqrt(
                float(self.rank * degree_count)
            )
            signs = (1,) if abs_order == 0 else (-1, 1)
            for sign in signs:
                indices_tuple = self.layout.signed_order_indices(abs_order, sign)
                indices = torch.tensor(
                    indices_tuple, dtype=torch.long, device=edges.values.device
                )
                degree_values = edges.values.index_select(1, indices)
                mixed = torch.einsum(
                    "eroi,rc,eic->eoc", kernel, factors, degree_values
                )
                replacement = degree_values + scale * mixed
                output = torch.index_copy(output, 1, indices, replacement)
        return SO2EdgeTensor(
            values=output,
            layout=edges.layout,
            topology_id=edges.topology_id,
            gauge_id=edges.gauge_id,
        )

