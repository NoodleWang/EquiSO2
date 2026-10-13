"""One Hessian-ready low-rank edge-local SO(2) message block."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from equiso2.graph.geometry import GeometryCache
from equiso2.graph.topology import EdgeTopology
from equiso2.irreps.layout import SO2Layout, SO3Layout, SO3NodeTensor
from equiso2.irreps.rotation import (
    aggregate_global_messages,
    lift_to_global,
    project_to_edge_gauge,
)

from .low_rank_so2 import LowRankSO2DegreeMixer
from .multi_focus import MultiFocusSO2Stack
from .smooth_attention import SmoothEnvelopeAttention


class _FunctionPreservingFeedbackLinear(nn.Linear):
    """Widen a scalar map without changing its initial function or RNG stream.

    The leading ``C x C`` block and bias are initialized exactly as an
    ``nn.Linear(C, C)``.  Added invariant-feedback columns are zero, and their
    initialization consumes no random numbers.  Consequently, models created
    with feedback disabled/enabled under the same seed have identical common
    parameters and identical initial scalar functions.
    """

    def reset_parameters(self) -> None:
        # ``nn.Linear.__init__`` has already assigned in/out_features and
        # allocated weight/bias before dispatching to this override.
        base_channels = self.out_features
        if self.in_features <= base_channels:
            raise ValueError("feedback linear must add at least one input block")
        scalar_weight = torch.empty(
            (self.out_features, base_channels),
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        nn.init.kaiming_uniform_(scalar_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.weight.zero_()
            self.weight[:, :base_channels].copy_(scalar_weight)
        if self.bias is not None:
            # Match nn.Linear(C, C), whose bias bound uses fan_in=C rather than
            # the widened input dimension.
            bound = 1.0 / math.sqrt(base_channels)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        base_channels = self.out_features
        scalar = F.linear(
            input[..., :base_channels],
            self.weight[:, :base_channels],
            self.bias,
        )
        feedback = F.linear(
            input[..., base_channels:],
            self.weight[:, base_channels:],
            None,
        )
        return scalar + feedback


class EquivariantNodeLinear(nn.Module):
    """Channel maps shared over every m component of a global degree."""

    def __init__(self, layout: SO3Layout, channels: int) -> None:
        super().__init__()
        self.layout = layout
        self.channels = int(channels)
        self.maps = nn.ModuleList(
            [nn.Linear(channels, channels, bias=(degree == 0)) for degree in range(layout.lmax + 1)]
        )

    def forward(self, nodes: SO3NodeTensor) -> SO3NodeTensor:
        if nodes.layout != self.layout or nodes.channels != self.channels:
            raise ValueError("node tensor does not match equivariant linear")
        pieces: list[Tensor] = []
        for degree, channel_map in enumerate(self.maps):
            degree_values = nodes.values[:, self.layout.degree_slice(degree), :]
            pieces.append(channel_map(degree_values))
        return SO3NodeTensor(torch.cat(pieces, dim=1), self.layout)


class EquivariantNodeGate(nn.Module):
    """Analytic norm-gated SO(3) nonlinearity with second derivatives.

    With ``invariant_scalar_feedback=True``, invariants of every non-scalar
    degree are concatenated into the scalar update.  ``sqrt_norm`` uses the
    regularized Euclidean norm, while ``squared_norm`` uses exactly
    ``sum_m x_lm**2``.  Both make the final non-scalar block outputs reachable
    from a scalar energy head without changing how non-scalar coefficients are
    gated.
    """

    def __init__(
        self,
        layout: SO3Layout,
        channels: int,
        epsilon: float = 1.0e-8,
        invariant_scalar_feedback: bool = False,
        invariant_scalar_feedback_epsilon: float = 1.0e-8,
        invariant_scalar_feedback_mode: str = "sqrt_norm",
    ) -> None:
        super().__init__()
        if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if not isinstance(invariant_scalar_feedback, bool):
            raise TypeError("invariant_scalar_feedback must be bool")
        if not isinstance(invariant_scalar_feedback_mode, str):
            raise TypeError("invariant_scalar_feedback_mode must be str")
        if invariant_scalar_feedback_mode not in {"sqrt_norm", "squared_norm"}:
            raise ValueError(
                "invariant_scalar_feedback_mode must be 'sqrt_norm' or "
                "'squared_norm'"
            )
        if (
            not math.isfinite(float(invariant_scalar_feedback_epsilon))
            or invariant_scalar_feedback_epsilon <= 0.0
        ):
            raise ValueError(
                "invariant_scalar_feedback_epsilon must be finite and positive"
            )
        self.layout = layout
        self.channels = int(channels)
        self.epsilon = float(epsilon)
        self.invariant_scalar_feedback = invariant_scalar_feedback
        self.invariant_scalar_feedback_mode = invariant_scalar_feedback_mode
        self.invariant_scalar_feedback_epsilon = float(
            invariant_scalar_feedback_epsilon
        )
        scalar_input_channels = (
            (layout.lmax + 1) * channels
            if invariant_scalar_feedback
            else channels
        )
        self.scalar_map = (
            _FunctionPreservingFeedbackLinear(scalar_input_channels, channels)
            if invariant_scalar_feedback
            else nn.Linear(scalar_input_channels, channels)
        )
        self.gates = nn.ModuleList(
            [nn.Identity()]
            + [nn.Linear(2 * channels, channels) for _ in range(1, layout.lmax + 1)]
        )

    def forward(self, nodes: SO3NodeTensor) -> SO3NodeTensor:
        scalar = nodes.values[:, self.layout.degree_slice(0), :].squeeze(1)
        if self.invariant_scalar_feedback:
            invariant_feedback: list[Tensor] = []
            for degree in range(1, self.layout.lmax + 1):
                values = nodes.values[:, self.layout.degree_slice(degree), :]
                squared_norm = values.square().sum(dim=1)
                if self.invariant_scalar_feedback_mode == "sqrt_norm":
                    invariant_feedback.append(
                        torch.sqrt(
                            squared_norm
                            + self.invariant_scalar_feedback_epsilon
                        )
                    )
                else:
                    # Deliberately no epsilon or square root: this alternative
                    # has finite zero-point curvature without the eps^-1/2
                    # scale of the regularized Euclidean norm.
                    invariant_feedback.append(squared_norm)
            scalar_input = torch.cat((scalar, *invariant_feedback), dim=1)
            scalar_context = torch.nn.functional.silu(
                self.scalar_map(scalar_input)
            )
        else:
            # Keep the default path operation-for-operation compatible with the
            # original checkpoint architecture.
            scalar_context = torch.nn.functional.silu(self.scalar_map(scalar))
        pieces: list[Tensor] = [scalar_context.unsqueeze(1)]
        for degree in range(1, self.layout.lmax + 1):
            values = nodes.values[:, self.layout.degree_slice(degree), :]
            # The pre-existing non-scalar gate keeps its original epsilon even
            # when feedback uses a larger regularizer.  Because the new
            # feedback columns are initialized to zero, this separation makes
            # every positive feedback epsilon exactly function-preserving at
            # initialization, not only the legacy 1e-8 value.
            norm = torch.sqrt(values.square().sum(dim=1) + self.epsilon)
            gate_input = torch.cat((scalar_context, norm), dim=1)
            gate = torch.sigmoid(self.gates[degree](gate_input)).unsqueeze(1)
            pieces.append(values * gate)
        return SO3NodeTensor(torch.cat(pieces, dim=1), self.layout)


class HessSO2Block(nn.Module):
    """A1 dynamic product -> multi-focus static stack -> smooth attention."""

    def __init__(
        self,
        node_layout: SO3Layout,
        edge_layout: SO2Layout,
        channels: int,
        radial_dim: int,
        radial_hidden: int,
        rank: int,
        num_foci: int,
        invariant_scalar_feedback: bool = False,
        invariant_scalar_feedback_epsilon: float = 1.0e-8,
        invariant_scalar_feedback_mode: str = "sqrt_norm",
    ) -> None:
        super().__init__()
        self.node_layout = node_layout
        self.edge_layout = edge_layout
        self.pre = EquivariantNodeLinear(node_layout, channels)
        self.skip = EquivariantNodeLinear(node_layout, channels)
        self.dynamic = LowRankSO2DegreeMixer(
            edge_layout, channels, radial_dim, rank, radial_hidden
        )
        self.static = MultiFocusSO2Stack(edge_layout, channels, num_foci)
        self.attention = SmoothEnvelopeAttention(
            edge_layout, channels, num_foci, radial_dim
        )
        self.gate = EquivariantNodeGate(
            node_layout,
            channels,
            invariant_scalar_feedback=invariant_scalar_feedback,
            invariant_scalar_feedback_epsilon=(
                invariant_scalar_feedback_epsilon
                if invariant_scalar_feedback
                else 1.0e-8
            ),
            invariant_scalar_feedback_mode=invariant_scalar_feedback_mode,
        )
        self.message_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        nodes: SO3NodeTensor,
        topology: EdgeTopology,
        geometry: GeometryCache,
    ) -> tuple[SO3NodeTensor, Tensor]:
        source_nodes = self.pre(nodes)
        local = project_to_edge_gauge(
            source_nodes, topology, geometry, self.edge_layout
        )
        local = self.dynamic(local, geometry.radial, geometry.envelope)
        local = self.static(local)
        local, weights = self.attention(
            local,
            geometry.radial,
            geometry.envelope,
            topology.dst,
            nodes.num_nodes,
        )
        global_edges = lift_to_global(local, geometry)
        messages = aggregate_global_messages(
            global_edges, topology.dst, nodes.num_nodes
        )
        residual = self.skip(nodes).values + self.message_scale * messages
        return self.gate(SO3NodeTensor(residual, self.node_layout)), weights
