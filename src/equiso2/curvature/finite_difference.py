"""Centered finite-difference force Jacobian/Hessian-vector products."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from . import CurvatureResult
from .sampler import ProbeBatch, _probe_values, _restore_probe_axis, validate_ptr


def _step_matrix(
    step_size: float | Tensor,
    *,
    n_probe: int,
    n_structure: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    step = torch.as_tensor(step_size, dtype=dtype, device=device)
    if step.ndim == 0:
        step = step.expand(n_probe, n_structure).clone()
    elif step.ndim == 1:
        if step.numel() == 1:
            step = step.expand(n_probe, n_structure).clone()
        elif step.numel() == n_probe and step.numel() == n_structure:
            raise ValueError(
                "Ambiguous one-dimensional step_size; use [P,1] for per-probe "
                "or [1,B] for per-structure values"
            )
        elif step.numel() == n_probe:
            step = step[:, None].expand(n_probe, n_structure)
        elif step.numel() == n_structure:
            step = step[None, :].expand(n_probe, n_structure)
        else:
            raise ValueError("One-dimensional step_size must have P or B entries")
    elif step.ndim == 2:
        try:
            step = torch.broadcast_to(step, (n_probe, n_structure))
        except RuntimeError as exc:
            raise ValueError("step_size is not broadcastable to [P,B]") from exc
    else:
        raise ValueError("step_size must be scalar, one-dimensional, or [P,B]")
    if not torch.all(torch.isfinite(step)) or torch.any(step <= 0):
        raise ValueError("All finite-difference step sizes must be finite and positive")
    return step


def _force_value(force_fn: Callable[[Tensor], Tensor], positions: Tensor) -> Tensor:
    force = force_fn(positions)
    if not isinstance(force, Tensor):
        raise TypeError("The force function must return a torch.Tensor")
    if force.numel() != positions.numel():
        raise ValueError("Force and positions must contain the same number of values")
    return force


def finite_difference_force_jvp(
    force_fn: Callable[[Tensor], Tensor],
    positions: Tensor,
    probes: Tensor | ProbeBatch,
    *,
    step_size: float | Tensor,
    ptr: Tensor | None = None,
    valid_mask: Tensor | None = None,
    assume_conservative: bool = False,
    topology_id: Any | None = None,
) -> CurvatureResult:
    """Approximate :math:`-J_Fv` with a centered force difference.

    For every probe this evaluates

    .. math::
       -[F(R+h v)-F(R-h v)]/(2h).

    A closure around ``force_fn`` must hold discrete topology and PBC images
    fixed, while the force implementation must rematerialize all continuous
    geometry at each perturbed coordinate (``docs/whf.md`` section 5.4).

    ``step_size`` can be scalar, per probe, per structure, or ``[P,B]``. Use an
    explicit two-dimensional shape when ``P == B`` to avoid ambiguity.
    """

    if not isinstance(positions, Tensor) or not positions.is_floating_point():
        raise TypeError("`positions` must be a floating torch.Tensor")
    probe_values, squeeze, probe_ptr, probe_mask = _probe_values(
        probes,
        expected_dof=positions.numel(),
        dtype=positions.dtype,
        device=positions.device,
    )
    if ptr is not None and probe_ptr is not None:
        ptr_arg = validate_ptr(ptr, n_atoms=positions.numel() // 3).to(positions.device)
        if not torch.equal(ptr_arg, probe_ptr):
            raise ValueError("Explicit ptr does not match ProbeBatch.ptr")
        ptr_t = ptr_arg
    elif ptr is not None:
        ptr_t = validate_ptr(ptr, n_atoms=positions.numel() // 3).to(positions.device)
    elif probe_ptr is not None:
        ptr_t = validate_ptr(probe_ptr, n_atoms=positions.numel() // 3).to(
            positions.device
        )
    else:
        if positions.numel() % 3:
            raise ValueError("Position tensor size must be divisible by three")
        ptr_t = torch.tensor(
            [0, positions.numel() // 3], dtype=torch.long, device=positions.device
        )

    n_probe = probe_values.shape[0]
    n_structure = ptr_t.numel() - 1
    if valid_mask is not None:
        explicit_mask = torch.as_tensor(
            valid_mask, dtype=torch.bool, device=positions.device
        )
        if explicit_mask.shape == (n_structure,):
            explicit_mask = explicit_mask[None, :].expand(n_probe, n_structure)
        if explicit_mask.shape != (n_probe, n_structure):
            raise ValueError("valid_mask must have shape [B] or [P,B]")
        structure_mask = (
            explicit_mask if probe_mask is None else explicit_mask & probe_mask
        )
    elif probe_mask is not None:
        structure_mask = probe_mask
    else:
        structure_mask = torch.ones(
            (n_probe, n_structure), dtype=torch.bool, device=positions.device
        )
    steps = _step_matrix(
        step_size,
        n_probe=n_probe,
        n_structure=n_structure,
        dtype=positions.dtype,
        device=positions.device,
    )

    coordinate_ptr = 3 * ptr_t
    actions: list[Tensor] = []
    for q, probe in enumerate(probe_values):
        if not torch.any(structure_mask[q]):
            actions.append(torch.zeros_like(probe))
            continue
        coordinate_steps = torch.empty_like(probe)
        for b in range(n_structure):
            start = int(coordinate_ptr[b].item())
            stop = int(coordinate_ptr[b + 1].item())
            coordinate_steps[start:stop] = steps[q, b]
        active_coordinate = torch.zeros_like(probe, dtype=torch.bool)
        for b in range(n_structure):
            if structure_mask[q, b]:
                start = int(coordinate_ptr[b].item())
                stop = int(coordinate_ptr[b + 1].item())
                active_coordinate[start:stop] = True
        perturbation = coordinate_steps * probe * active_coordinate.to(probe.dtype)
        plus = positions + perturbation.reshape_as(positions)
        minus = positions - perturbation.reshape_as(positions)
        force_plus = _force_value(force_fn, plus).reshape(-1)
        force_minus = _force_value(force_fn, minus).reshape(-1)
        action = -(force_plus - force_minus) / (2.0 * coordinate_steps)
        action = torch.where(active_coordinate, action, torch.zeros_like(action))
        actions.append(action)
    stacked = torch.stack(actions, dim=0) if actions else probe_values.clone()
    result_probe = _restore_probe_axis(probe_values, squeeze)
    restored_action = _restore_probe_axis(stacked, squeeze)
    return CurvatureResult(
        method="conservative_fd_hvp" if assume_conservative else "force_fd_jvp",
        probe=result_probe,
        minus_Jv=restored_action,
        Hv=restored_action if assume_conservative else None,
        step_size=steps,
        ptr=ptr_t,
        topology_id=topology_id,
        valid_mask=structure_mask,
    )


def finite_difference_hvp(
    force_fn: Callable[[Tensor], Tensor],
    positions: Tensor,
    probes: Tensor | ProbeBatch,
    *,
    step_size: float | Tensor,
    ptr: Tensor | None = None,
    valid_mask: Tensor | None = None,
    topology_id: Any | None = None,
) -> CurvatureResult:
    """Centered FD-HVP for a force explicitly known to be conservative."""

    return finite_difference_force_jvp(
        force_fn,
        positions,
        probes,
        step_size=step_size,
        ptr=ptr,
        valid_mask=valid_mask,
        assume_conservative=True,
        topology_id=topology_id,
    )


__all__ = ["finite_difference_force_jvp", "finite_difference_hvp"]
