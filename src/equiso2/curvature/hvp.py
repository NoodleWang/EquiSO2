"""Exact energy Hessian products and forward-oriented force Jacobian products."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from . import CurvatureResult
from .sampler import ProbeBatch, _probe_values, _restore_probe_axis
from .vjp import _coordinate_input


def exact_energy_hvp(
    energy_fn: Callable[[Tensor], Tensor],
    positions: Tensor,
    probes: Tensor | ProbeBatch,
    *,
    create_graph: bool = False,
    topology_id: Any | None = None,
) -> CurvatureResult:
    """Apply the exact Hessian of a scalar energy to one or more probes.

    ``energy_fn`` may return a scalar or one energy per independent structure;
    the values are summed before differentiation. Backpropagating a loss through
    this result with ``create_graph=True`` generally requires mixed third-order
    derivatives, which is why the design uses this routine primarily as a small
    FP64 oracle (``docs/whf.md`` section 5.3).
    """

    coordinate = _coordinate_input(positions)
    probe_values, squeeze, ptr, structure_mask = _probe_values(
        probes,
        expected_dof=coordinate.numel(),
        dtype=coordinate.dtype,
        device=coordinate.device,
    )
    energy = energy_fn(coordinate)
    if not isinstance(energy, Tensor):
        raise TypeError("The energy function must return a torch.Tensor")
    if not energy.requires_grad:
        raise ValueError("Energy is not connected to the coordinate graph")
    (gradient,) = torch.autograd.grad(
        energy.sum(),
        coordinate,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    actions: list[Tensor] = []
    for q, probe in enumerate(probe_values):
        # The first-order graph is also needed when a create_graph=True result is
        # later differentiated with respect to model parameters.
        keep_graph = True if (create_graph or q + 1 < probe_values.shape[0]) else False
        (h_v,) = torch.autograd.grad(
            gradient,
            coordinate,
            grad_outputs=probe.reshape_as(gradient),
            create_graph=create_graph,
            retain_graph=keep_graph,
            allow_unused=False,
        )
        actions.append(h_v.reshape(-1))
    stacked = torch.stack(actions, dim=0) if actions else probe_values.clone()
    result_probe = _restore_probe_axis(probe_values, squeeze)
    return CurvatureResult(
        method="exact_energy_hvp",
        probe=result_probe,
        Hv=_restore_probe_axis(stacked, squeeze),
        ptr=ptr,
        topology_id=topology_id,
        valid_mask=structure_mask,
    )


def minus_force_jacobian_vector(
    force_fn: Callable[[Tensor], Tensor],
    positions: Tensor,
    probes: Tensor | ProbeBatch,
    *,
    create_graph: bool = False,
    strict: bool = False,
    topology_id: Any | None = None,
) -> CurvatureResult:
    """Compute the forward-oriented action :math:`-J_F v` with a JVP.

    Unlike a VJP, this is the orientation needed by a centered force difference.
    It remains named ``minus_Jv`` for a direct force; callers must not relabel it
    ``Hv`` unless conservativity has been established.
    """

    coordinate = _coordinate_input(positions)
    probe_values, squeeze, ptr, structure_mask = _probe_values(
        probes,
        expected_dof=coordinate.numel(),
        dtype=coordinate.dtype,
        device=coordinate.device,
    )

    def checked_force(x: Tensor) -> Tensor:
        value = force_fn(x)
        if not isinstance(value, Tensor):
            raise TypeError("The force function must return a torch.Tensor")
        if value.numel() != x.numel():
            raise ValueError("Force and positions must have the same number of values")
        return value

    actions: list[Tensor] = []
    for probe in probe_values:
        _, j_v = torch.autograd.functional.jvp(
            checked_force,
            coordinate,
            probe.reshape_as(coordinate),
            create_graph=create_graph,
            strict=strict,
        )
        actions.append(-j_v.reshape(-1))
    stacked = torch.stack(actions, dim=0) if actions else probe_values.clone()
    result_probe = _restore_probe_axis(probe_values, squeeze)
    return CurvatureResult(
        method="force_jvp",
        probe=result_probe,
        minus_Jv=_restore_probe_axis(stacked, squeeze),
        ptr=ptr,
        topology_id=topology_id,
        valid_mask=structure_mask,
    )


__all__ = ["exact_energy_hvp", "minus_force_jacobian_vector"]
