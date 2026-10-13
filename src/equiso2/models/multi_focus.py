"""Narrow static SO(2) stacks with smooth invariant focus competition."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from equiso2.irreps.layout import SO2EdgeTensor, SO2Layout


class MultiFocusSO2Stack(nn.Module):
    r"""Apply independent narrow SO(2) residual maps and soft focus weights.

    This implements ``whf.md``, Section 4.5.  Each signed m partner is passed
    through the same map; normalization is joint over the pair.  Focus scores
    use only the local ``(l,m)=(0,0)`` coefficient, so the weights are invariant.
    """

    def __init__(
        self,
        layout: SO2Layout,
        channels: int,
        num_foci: int,
        temperature: float = 1.0,
        norm_epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if num_foci <= 0 or channels % num_foci != 0:
            raise ValueError("channels must be divisible by a positive num_foci")
        if temperature <= 0.0 or norm_epsilon <= 0.0:
            raise ValueError("temperature and norm_epsilon must be positive")
        self.layout = layout
        self.channels = int(channels)
        self.num_foci = int(num_foci)
        self.focus_channels = channels // num_foci
        self.temperature = float(temperature)
        self.norm_epsilon = float(norm_epsilon)
        self.linears = nn.ModuleDict()
        self.residual_scales = nn.ParameterDict()
        self.focus_scores = nn.ModuleList()
        for focus in range(num_foci):
            for abs_order in range(layout.mmax + 1):
                degree_count = layout.lmax - abs_order + 1
                width = degree_count * self.focus_channels
                key = f"f{focus}_m{abs_order}"
                self.linears[key] = nn.Linear(width, width, bias=False)
                self.residual_scales[key] = nn.Parameter(torch.tensor(0.1))
            self.focus_scores.append(
                nn.Sequential(
                    nn.Linear(self.focus_channels, self.focus_channels),
                    nn.SiLU(),
                    nn.Linear(self.focus_channels, 1, bias=False),
                )
            )

    def _focus_update(self, values: Tensor, focus: int) -> Tensor:
        edge_count = values.shape[0]
        output = values
        for abs_order in range(self.layout.mmax + 1):
            signs = (1,) if abs_order == 0 else (-1, 1)
            gathered: list[tuple[Tensor, Tensor]] = []
            squared_sum = values.new_zeros((edge_count,))
            element_count = 0
            for sign in signs:
                index = torch.tensor(
                    self.layout.signed_order_indices(abs_order, sign),
                    dtype=torch.long,
                    device=values.device,
                )
                part = values.index_select(1, index)
                gathered.append((index, part))
                squared_sum = squared_sum + part.square().sum(dim=(1, 2))
                element_count += part.shape[1] * part.shape[2]
            rms = torch.sqrt(
                squared_sum / float(element_count) + self.norm_epsilon
            )
            key = f"f{focus}_m{abs_order}"
            for index, part in gathered:
                normalized = part / rms.reshape(edge_count, 1, 1)
                # ``reshape(0, -1)`` is ambiguous in PyTorch.  The feature
                # width is known from the SO(2) block, so keep it explicit for
                # valid zero-edge graphs such as a single isolated atom.
                flat_width = part.shape[1] * part.shape[2]
                flat = normalized.reshape(edge_count, flat_width)
                delta = self.linears[key](flat).reshape_as(part)
                replacement = part + self.residual_scales[key] * delta
                output = torch.index_copy(output, 1, index, replacement)
        return output

    def forward(self, edges: SO2EdgeTensor) -> SO2EdgeTensor:
        if edges.layout != self.layout or edges.channels != self.channels:
            raise ValueError("edge tensor does not match multi-focus stack")
        scalar_index = self.layout.compact_index(0, 0)
        updated_foci: list[Tensor] = []
        scores: list[Tensor] = []
        for focus in range(self.num_foci):
            begin = focus * self.focus_channels
            end = begin + self.focus_channels
            focus_values = edges.values[:, :, begin:end]
            updated = self._focus_update(focus_values, focus)
            updated_foci.append(updated)
            scalar = updated[:, scalar_index, :]
            scores.append(self.focus_scores[focus](scalar))
        logits = torch.cat(scores, dim=1) / self.temperature
        probabilities = torch.softmax(logits, dim=1)
        weighted: list[Tensor] = []
        for focus, values in enumerate(updated_foci):
            weight = probabilities[:, focus].reshape(edges.num_edges, 1, 1)
            weighted.append(values * weight)
        return SO2EdgeTensor(
            values=torch.cat(weighted, dim=2),
            layout=edges.layout,
            topology_id=edges.topology_id,
            gauge_id=edges.gauge_id,
        )
