"""Packed blockwise controller for the EH/weak-force Sobolev ablation.

This module is intentionally separate from
``blockwise_derivative_balance_v1``.  The registered E/F/H controller gives
Force and Hessian equal, unit-normalized contributions and therefore cannot
represent either strict E/H training or a weak Force anchor.

The rule implemented here is, on every derivative-reachable parameter block,

``g = project_H(g_E) + rho_F ||g_E|| unit(g_F)
                        + rho_H ||g_E|| unit(g_H)``.

Only Hessian is a hard non-conflict task: ``project_H`` projects the energy
direction onto the Hessian non-conflict half-space and the completed update is
checked against the Hessian gradient.  Force is deliberately a *soft* anchor;
its dot product is reported but never used as a gate.  This makes the weak
arm continuous as ``rho_F -> 0``.  At exactly ``rho_F == 0`` the caller must
provide an all-``None`` Force gradient tuple, proving that strict E/H training
does not construct or consume a Force-label parameter gradient.

Inputs are packed to one synthetic tensor per coarse parameter group before
the balancing algebra is applied, then restored to their original layout.
The legacy controller and its schema are not modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from .blockwise_derivative_balance_packed_ablation_v1 import (
    _as_aligned_gradient_tuples,
    _batched_finite_and_reachable_flags,
    _pack_component,
    _reference_for_index,
)
from .blockwise_derivative_balance_v1 import (
    BlockwiseBalanceResult,
    _cosine,
    _is_reachable,
    _linear_combination,
    _project_energy_to_halfspaces,
    _projection_tolerance,
    _scaled_tuple,
    _stable_dot,
    _stable_norm,
    _unit_direction,
)


SCHEMA_VERSION = "equiso2-blockwise-sobolev-balance-packed-v1"
GradientTuple = tuple[Tensor | None, ...]


@dataclass(frozen=True)
class _PackedGroup:
    name: str
    packed_index: int
    original_indices: tuple[int, ...]
    active_indices: tuple[int, ...]
    dead_indices: tuple[int, ...]
    shapes: tuple[torch.Size, ...]
    scalar_counts: tuple[int, ...]


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(value: Any, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _split_component(
    packed: Tensor | None,
    group: _PackedGroup,
    output: list[Tensor | None],
) -> None:
    if packed is None:
        return
    cursor = 0
    for index, shape, scalar_count in zip(
        group.active_indices,
        group.shapes,
        group.scalar_counts,
        strict=True,
    ):
        output[index] = packed[cursor : cursor + scalar_count].reshape(shape).detach()
        cursor += scalar_count
    if cursor != packed.numel():
        raise RuntimeError("packed controller output has an invalid scalar count")


def _no_projection_audit(dtype: torch.dtype) -> dict[str, Any]:
    tolerance = _projection_tolerance(dtype)
    return {
        "active_set": "none_no_hessian",
        "projection_applied": False,
        "projection_distance_l2": 0.0,
        "pre_renormalization_norm": 1.0,
        "renormalization_factor": 1.0,
        "post_unit_task_dots": {},
        "numerical_tolerance": tolerance,
        "post_numerical_tolerance": tolerance,
    }


def _full_gradient_metrics(
    combined: GradientTuple,
    force: GradientTuple,
    hessian: GradientTuple,
    energy: GradientTuple,
    *,
    force_enabled: bool,
    dtype: torch.dtype,
) -> dict[str, float | bool | None]:
    """Compute full-layout metrics in one explicit device-to-host transfer."""

    references = tuple(
        next(
            (
                value
                for value in (
                    combined[index],
                    energy[index],
                    force[index],
                    hessian[index],
                )
                if value is not None
            ),
            None,
        )
        for index in range(len(combined))
    )

    def flatten(values: GradientTuple) -> Tensor:
        pieces = [
            (
                torch.zeros_like(reference)
                if values[index] is None
                else values[index].detach()
            ).reshape(-1)
            for index, reference in enumerate(references)
            if reference is not None
        ]
        if not pieces:
            raise RuntimeError("global gradient metrics have no tensor reference")
        return torch.cat(pieces).to(dtype=torch.float64)

    def stable_norm(vector: Tensor) -> Tensor:
        maximum = vector.abs().amax()
        safe_maximum = maximum.clamp_min(torch.finfo(torch.float64).tiny)
        return maximum * torch.sqrt(torch.square(vector / safe_maximum).sum())

    combined_vector = flatten(combined)
    hessian_vector = flatten(hessian)
    combined_norm = stable_norm(combined_vector)
    hessian_norm = stable_norm(hessian_vector)
    hessian_dot = torch.dot(combined_vector, hessian_vector)
    values = [combined_norm, hessian_norm, hessian_dot]
    if force_enabled:
        force_vector = flatten(force)
        force_norm = stable_norm(force_vector)
        force_dot = torch.dot(combined_vector, force_vector)
        values.extend((force_norm, force_dot))
    host = torch.stack(values).to(device="cpu").tolist()
    combined_norm_value, hessian_norm_value, hessian_dot_value = (
        float(value) for value in host[:3]
    )
    if combined_norm_value <= 0.0 or hessian_norm_value <= 0.0:
        raise RuntimeError("assigned and Hessian gradients must be globally non-zero")
    tolerance = _projection_tolerance(dtype)
    hessian_threshold = (
        -tolerance * combined_norm_value * hessian_norm_value
    )
    if hessian_dot_value < hessian_threshold:
        raise FloatingPointError("assigned global gradient conflicts with Hessian")

    result: dict[str, float | bool | None] = {
        "global_assigned_gradient_l2": combined_norm_value,
        "global_hessian_gradient_l2": hessian_norm_value,
        "global_assigned_hessian_dot": hessian_dot_value,
        "global_assigned_hessian_cosine": hessian_dot_value
        / (combined_norm_value * hessian_norm_value),
        "global_hessian_nonconflict_passed": True,
        "global_force_gradient_l2": None,
        "global_assigned_force_dot": None,
        "global_assigned_force_cosine": None,
        "global_soft_force_compatible": None,
    }
    if force_enabled:
        force_norm_value, force_dot_value = (float(value) for value in host[3:])
        if force_norm_value <= 0.0:
            raise RuntimeError("enabled Force gradient must be globally non-zero")
        result.update(
            {
                "global_force_gradient_l2": force_norm_value,
                "global_assigned_force_dot": force_dot_value,
                "global_assigned_force_cosine": force_dot_value
                / (combined_norm_value * force_norm_value),
                # Diagnostic only: this boolean is intentionally not a gate.
                "global_soft_force_compatible": force_dot_value
                >= -tolerance * combined_norm_value * force_norm_value,
            }
        )
    return result


def _balance_packed_group(
    *,
    group: str,
    energy: Tensor,
    force: Tensor | None,
    hessian: Tensor,
    force_target_ratio: float,
    hessian_target_ratio: float,
) -> tuple[
    Tensor,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor,
    dict[str, Any],
]:
    """Apply the hard-H/soft-F rule to one packed coarse group."""

    energy_tuple: GradientTuple = (energy,)
    force_tuple: GradientTuple = (force,)
    hessian_tuple: GradientTuple = (hessian,)
    indices = (0,)
    dtype = energy.dtype
    energy_norm = _stable_norm(energy_tuple, indices)
    if energy_norm == 0.0:
        raise RuntimeError(
            f"zero energy gradient on derivative-reachable block {group!r}"
        )
    energy_unit = _unit_direction(
        energy_tuple,
        indices,
        energy_tuple,
        name=f"energy/{group}",
    )
    hessian_active = _is_reachable(hessian)
    force_active = force_target_ratio > 0.0 and _is_reachable(force)
    if not hessian_active and not force_active:
        raise RuntimeError(f"packed block {group!r} has no enabled derivative task")

    hessian_unit = (
        _unit_direction(
            hessian_tuple,
            indices,
            energy_tuple,
            name=f"Hessian/{group}",
        )
        if hessian_active
        else None
    )
    force_unit = (
        _unit_direction(
            force_tuple,
            indices,
            energy_tuple,
            name=f"force/{group}",
        )
        if force_active
        else None
    )

    if hessian_unit is None:
        safe_energy = energy_tuple
        projection_audit = _no_projection_audit(dtype)
    else:
        safe_energy_unit, projection_audit = _project_energy_to_halfspaces(
            energy_unit,
            None,
            hessian_unit,
            indices,
            dtype=dtype,
        )
        safe_energy = (
            _scaled_tuple(safe_energy_unit, indices, energy_norm)
            if projection_audit["projection_applied"]
            else energy_tuple
        )

    force_part: GradientTuple = (
        _scaled_tuple(
            force_unit,
            indices,
            force_target_ratio * energy_norm,
        )
        if force_unit is not None
        else (None,)
    )
    hessian_part: GradientTuple = (
        _scaled_tuple(
            hessian_unit,
            indices,
            hessian_target_ratio * energy_norm,
        )
        if hessian_unit is not None
        else (None,)
    )
    derivative_part = _linear_combination(
        energy_tuple,
        indices,
        ((1.0, force_part), (1.0, hessian_part)),
    )
    if _stable_norm(derivative_part, indices) == 0.0:
        raise RuntimeError(f"derivative contributions cancel exactly in {group!r}")
    combined = _linear_combination(
        energy_tuple,
        indices,
        ((1.0, safe_energy), (1.0, derivative_part)),
    )
    combined_norm = _stable_norm(combined, indices)
    if combined_norm == 0.0:
        raise RuntimeError(f"zero assigned gradient in {group!r}")
    tolerance = _projection_tolerance(dtype)
    hessian_dot: float | None = None
    hessian_cosine: float | None = None
    if hessian_active:
        hessian_norm = _stable_norm(hessian_tuple, indices)
        hessian_dot = _stable_dot(combined, hessian_tuple, indices)
        if hessian_dot < -tolerance * combined_norm * hessian_norm:
            raise FloatingPointError(
                f"assigned block gradient conflicts with Hessian in {group!r}"
            )
        hessian_cosine = hessian_dot / (combined_norm * hessian_norm)

    force_dot: float | None = None
    force_cosine: float | None = None
    force_compatible: bool | None = None
    if force_active:
        force_norm = _stable_norm(force_tuple, indices)
        force_dot = _stable_dot(combined, force_tuple, indices)
        force_cosine = force_dot / (combined_norm * force_norm)
        force_compatible = force_dot >= -tolerance * combined_norm * force_norm

    projected_energy_norm = _stable_norm(safe_energy, indices)
    if abs(projected_energy_norm - energy_norm) > tolerance * max(1.0, energy_norm):
        raise FloatingPointError("energy block norm was not preserved")
    force_contribution_norm = (
        None if force_unit is None else _stable_norm(force_part, indices)
    )
    hessian_contribution_norm = (
        None if hessian_unit is None else _stable_norm(hessian_part, indices)
    )
    block_audit = {
        "group": group,
        "status": "balanced",
        "controller_semantics": "hard_hessian_soft_force_anchor_v1",
        "energy_gradient_l2_before": energy_norm,
        "energy_gradient_l2_after_projection": projected_energy_norm,
        "force_gradient_l2": (
            None if not force_active else _stable_norm(force_tuple, indices)
        ),
        "hessian_gradient_l2": (
            None if not hessian_active else _stable_norm(hessian_tuple, indices)
        ),
        "force_target_ratio": force_target_ratio,
        "hessian_target_ratio": hessian_target_ratio,
        "force_contribution_l2": force_contribution_norm,
        "hessian_contribution_l2": hessian_contribution_norm,
        "derivative_contribution_l2": _stable_norm(derivative_part, indices),
        "energy_force_cosine_before": (
            None if not force_active else _cosine(energy_tuple, force_tuple, indices)
        ),
        "energy_hessian_cosine_before": (
            None
            if not hessian_active
            else _cosine(energy_tuple, hessian_tuple, indices)
        ),
        "assigned_force_dot": force_dot,
        "assigned_force_cosine": force_cosine,
        "soft_force_compatible": force_compatible,
        "force_is_hard_gate": False,
        "assigned_hessian_dot": hessian_dot,
        "assigned_hessian_cosine": hessian_cosine,
        "hessian_nonconflict_passed": True if hessian_active else None,
        "hessian_is_hard_gate": True,
        "assigned_gradient_l2": combined_norm,
        "projection": projection_audit,
    }
    assert combined[0] is not None
    assert safe_energy[0] is not None
    assert derivative_part[0] is not None
    return (
        combined[0],
        safe_energy[0],
        force_part[0],
        hessian_part[0],
        derivative_part[0],
        block_audit,
    )


def balance_blockwise_sobolev_gradients_packed(
    parameter_names: Sequence[str],
    energy_gradients: Sequence[Tensor | None],
    force_gradients: Sequence[Tensor | None],
    hessian_gradients: Sequence[Tensor | None],
    *,
    parameter_groups: Mapping[str, str],
    force_target_ratio: float,
    hessian_target_ratio: float,
) -> BlockwiseBalanceResult:
    """Return a packed hard-Hessian/soft-Force blockwise gradient.

    ``force_target_ratio=0`` is the strict E/H arm and requires every Force
    gradient entry to be ``None``.  A positive Force ratio enables a soft
    unit-direction contribution but does not add Force to the projection or
    to any pass/fail gate.  ``hessian_target_ratio`` must be positive, and
    Hessian non-conflict is enforced blockwise and globally.
    """

    rho_force = _finite_nonnegative(
        force_target_ratio, name="force_target_ratio"
    )
    rho_hessian = _finite_positive(
        hessian_target_ratio, name="hessian_target_ratio"
    )
    (
        names,
        energy,
        force,
        hessian,
        grouped_indices,
        device,
        dtype,
    ) = _as_aligned_gradient_tuples(
        parameter_names,
        energy_gradients,
        force_gradients,
        hessian_gradients,
        parameter_groups,
    )
    if rho_force == 0.0 and any(value is not None for value in force):
        raise ValueError(
            "strict E/H requires an all-None Force gradient tuple"
        )
    flags = _batched_finite_and_reachable_flags(
        (energy, force, hessian), device=device
    )
    for component_name, component_flags in zip(
        ("energy", "force", "hessian"), flags, strict=True
    ):
        if not all(finite for finite, _ in component_flags):
            raise FloatingPointError(f"non-finite {component_name} gradient")
    force_reachable = tuple(reachable for _, reachable in flags[1])
    hessian_reachable = tuple(reachable for _, reachable in flags[2])
    if not any(hessian_reachable):
        raise RuntimeError("zero global Hessian gradient prevents balancing")
    if rho_force > 0.0 and not any(force_reachable):
        raise RuntimeError("zero global Force gradient prevents weak anchoring")

    packed_energy: list[Tensor] = []
    packed_force: list[Tensor | None] = []
    packed_hessian: list[Tensor] = []
    packed_groups: list[_PackedGroup] = []
    for group, original_indices in grouped_indices.items():
        active_indices = tuple(
            index
            for index in original_indices
            if hessian_reachable[index]
            or (rho_force > 0.0 and force_reachable[index])
        )
        active_set = set(active_indices)
        dead_indices = tuple(
            index for index in original_indices if index not in active_set
        )
        if not active_indices:
            continue
        references = {
            index: _reference_for_index(index, energy, force, hessian)
            for index in active_indices
        }
        packed_index = len(packed_groups)
        packed_energy.append(
            _pack_component(energy, active_indices, references)
        )
        packed_force.append(
            _pack_component(force, active_indices, references)
            if rho_force > 0.0
            else None
        )
        packed_hessian.append(
            _pack_component(hessian, active_indices, references)
        )
        packed_groups.append(
            _PackedGroup(
                name=group,
                packed_index=packed_index,
                original_indices=original_indices,
                active_indices=active_indices,
                dead_indices=dead_indices,
                shapes=tuple(references[index].shape for index in active_indices),
                scalar_counts=tuple(
                    references[index].numel() for index in active_indices
                ),
            )
        )
    if not packed_groups:
        raise RuntimeError("no derivative-reachable parameter block")

    packed_outputs: list[
        tuple[
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            Tensor,
            dict[str, Any],
        ]
    ] = []
    for packed_group in packed_groups:
        index = packed_group.packed_index
        packed_outputs.append(
            _balance_packed_group(
                group=packed_group.name,
                energy=packed_energy[index],
                force=packed_force[index],
                hessian=packed_hessian[index],
                force_target_ratio=rho_force,
                hessian_target_ratio=rho_hessian,
            )
        )

    output_fields: list[list[Tensor | None]] = [
        [None] * len(names) for _ in range(5)
    ]
    combined, projected, force_part, hessian_part, derivative_part = output_fields
    all_active = {
        index for packed_group in packed_groups for index in packed_group.active_indices
    }
    for index in range(len(names)):
        if index not in all_active and energy[index] is not None:
            combined[index] = energy[index].detach()
            projected[index] = energy[index].detach()

    block_audits: list[dict[str, Any]] = []
    packed_by_name = {group.name: group for group in packed_groups}
    for group, original_indices in grouped_indices.items():
        packed_group = packed_by_name.get(group)
        if packed_group is None:
            block_audits.append(
                {
                    "group": group,
                    "status": "derivative_dead",
                    "original_parameter_tensor_count": len(original_indices),
                    "derivative_active_tensor_count": 0,
                    "derivative_dead_tensor_count": len(original_indices),
                    "packed_scalar_count": 0,
                    "derivative_dead_energy_preserved": True,
                    "controller_semantics": "hard_hessian_soft_force_anchor_v1",
                }
            )
            continue
        packed_output = packed_outputs[packed_group.packed_index]
        for packed_value, output in zip(
            packed_output[:5], output_fields, strict=True
        ):
            _split_component(packed_value, packed_group, output)
        projection = packed_output[5]["projection"]
        if not projection["projection_applied"]:
            for index in packed_group.active_indices:
                if energy[index] is None:
                    projected[index] = None
        block_audit = dict(packed_output[5])
        block_audit.update(
            {
                "original_parameter_tensor_count": len(original_indices),
                "derivative_active_tensor_count": len(
                    packed_group.active_indices
                ),
                "derivative_dead_tensor_count": len(packed_group.dead_indices),
                "packed_scalar_count": sum(packed_group.scalar_counts),
                "derivative_dead_energy_preserved": all(
                    energy[index] is None
                    or torch.equal(combined[index], energy[index].detach())
                    for index in packed_group.dead_indices
                ),
            }
        )
        block_audits.append(block_audit)

    combined_tuple = tuple(combined)
    global_metrics = _full_gradient_metrics(
        combined_tuple,
        force,
        hessian,
        energy,
        force_enabled=rho_force > 0.0,
        dtype=dtype,
    )
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "controller_semantics": "hard_hessian_soft_force_anchor_v1",
        "force_target_ratio": rho_force,
        "hessian_target_ratio": rho_hessian,
        "force_enabled": rho_force > 0.0,
        "force_is_hard_gate": False,
        "hessian_is_hard_gate": True,
        "strict_eh": rho_force == 0.0,
        "original_parameter_tensor_count": len(names),
        "packed_parameter_tensor_count": len(packed_groups),
        "original_group_count": len(grouped_indices),
        "packed_group_count": len(packed_groups),
        "derivative_active_tensor_count": len(all_active),
        "derivative_dead_tensor_count": len(names) - len(all_active),
        "packed_scalar_count": sum(
            scalar_count
            for packed_group in packed_groups
            for scalar_count in packed_group.scalar_counts
        ),
        "explicit_flag_host_sync_count": 1,
        "explicit_metric_host_sync_count": 1,
        "explicit_total_host_sync_count": 2,
        "numerical_tolerance": _projection_tolerance(dtype),
        **global_metrics,
        "blocks": block_audits,
    }
    return BlockwiseBalanceResult(
        combined_gradients=combined_tuple,
        projected_energy_gradients=tuple(projected),
        force_contributions=tuple(force_part),
        hessian_contributions=tuple(hessian_part),
        derivative_contributions=tuple(derivative_part),
        audit=audit,
    )


__all__ = [
    "SCHEMA_VERSION",
    "balance_blockwise_sobolev_gradients_packed",
]
