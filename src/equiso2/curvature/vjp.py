"""Reverse-mode force Jacobian products.

For a direct force :math:`F(R)`, ``autograd.grad(F, R, grad_outputs=v)``
returns :math:`J_F^T v`. This module always attaches the required minus sign
and calls the result ``minus_JT_v``; it never labels it ``Hv`` merely because a
reference Hessian happens to be symmetric.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from . import CurvatureResult
from .sampler import ProbeBatch, _probe_values, _restore_probe_axis


def _coordinate_input(positions: Tensor) -> Tensor:
    if not isinstance(positions, Tensor):
        raise TypeError("`positions` must be a torch.Tensor")
    if not positions.is_floating_point():
        raise TypeError("`positions` must have a floating dtype")
    if positions.requires_grad:
        return positions
    # A new coordinate leaf is necessary for convenience. This does not detach
    # model parameters captured by a callable evaluated afterwards.
    return positions.detach().requires_grad_(True)


def _force_tensor(force: Tensor, positions: Tensor) -> Tensor:
    if not isinstance(force, Tensor):
        raise TypeError("The force function must return a torch.Tensor")
    if force.numel() != positions.numel():
        raise ValueError("Force and position tensors must contain the same number of values")
    if not force.requires_grad:
        raise ValueError("Force is not connected to a differentiable coordinate graph")
    return force


def minus_force_jacobian_transpose_vector(
    force: Tensor,
    positions: Tensor,
    probes: Tensor | ProbeBatch,
    *,
    create_graph: bool = True,
    retain_graph: bool | None = None,
    topology_id: Any | None = None,
) -> CurvatureResult:
    """Compute :math:`-J_F^T v` for precomputed differentiable forces.

    Parameters follow ``docs/whf.md`` section 5.2. ``create_graph=True`` is the
    training setting: it preserves mixed coordinate--parameter derivatives so a
    loss on the returned action can update model parameters.
    """

    positions = _coordinate_input(positions)
    force = _force_tensor(force, positions)
    probe_values, squeeze, ptr, structure_mask = _probe_values(
        probes,
        expected_dof=positions.numel(),
        dtype=positions.dtype,
        device=positions.device,
    )
    keep_last = create_graph if retain_graph is None else retain_graph
    actions: list[Tensor] = []
    for q, probe in enumerate(probe_values):
        keep_graph = True if q + 1 < probe_values.shape[0] else keep_last
        (jt_v,) = torch.autograd.grad(
            outputs=force,
            inputs=positions,
            grad_outputs=probe.reshape_as(force),
            create_graph=create_graph,
            retain_graph=keep_graph,
            allow_unused=False,
        )
        actions.append(-jt_v.reshape(-1))
    stacked = torch.stack(actions, dim=0) if actions else probe_values.clone()
    result_probe = _restore_probe_axis(probe_values, squeeze)
    return CurvatureResult(
        method="direct_vjp",
        probe=result_probe,
        minus_JT_v=_restore_probe_axis(stacked, squeeze),
        ptr=ptr,
        topology_id=topology_id,
        valid_mask=structure_mask,
    )


def direct_force_vjp(
    force_fn: Callable[[Tensor], Tensor],
    positions: Tensor,
    probes: Tensor | ProbeBatch,
    *,
    create_graph: bool = True,
    retain_graph: bool | None = None,
    topology_id: Any | None = None,
) -> CurvatureResult:
    """Evaluate a direct-force callable and return its oriented VJP action."""

    coordinate = _coordinate_input(positions)
    force = force_fn(coordinate)
    return minus_force_jacobian_transpose_vector(
        force,
        coordinate,
        probes,
        create_graph=create_graph,
        retain_graph=retain_graph,
        topology_id=topology_id,
    )


__all__ = ["direct_force_vjp", "minus_force_jacobian_transpose_vector"]
