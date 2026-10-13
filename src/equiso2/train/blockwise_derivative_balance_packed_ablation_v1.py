"""Packed-gradient adapter for the blockwise derivative controller.

This is an isolated performance-ablation implementation.  It preserves the
published v1 balancing rule by concatenating all derivative-reachable tensors
in a coarse parameter group, calling the existing controller once per packed
group, and then restoring the original tensor layout.  It does not replace or
modify :mod:`equiso2.train.blockwise_derivative_balance_v1`.

The adapter obtains finite/reachable flags with one device-to-host transfer.
The wrapped controller can therefore perform its scalar diagnostics at coarse
group granularity rather than once per original parameter tensor.  All packed
inputs are detached, so every returned gradient is outside the autograd graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from equiso2.train.blockwise_derivative_balance_v1 import (
    BlockwiseBalanceResult,
    balance_blockwise_derivative_gradients,
    group_parameter_indices,
)


SCHEMA_VERSION = "equiso2-blockwise-derivative-balance-packed-ablation-v1"
INDEPENDENT_TASK_RATIO_SCHEMA_VERSION = (
    "equiso2-blockwise-derivative-balance-packed-independent-task-ratios-v1"
)
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


def _as_aligned_gradient_tuples(
    parameter_names: Sequence[str],
    energy_gradients: Sequence[Tensor | None],
    force_gradients: Sequence[Tensor | None],
    hessian_gradients: Sequence[Tensor | None],
    parameter_groups: Mapping[str, str],
) -> tuple[
    tuple[str, ...],
    GradientTuple,
    GradientTuple,
    GradientTuple,
    dict[str, tuple[int, ...]],
    torch.device,
    torch.dtype,
]:
    """Validate host-visible metadata without synchronizing tensor values."""

    names = tuple(parameter_names)
    groups = group_parameter_indices(names, parameter_groups)
    components = (
        tuple(energy_gradients),
        tuple(force_gradients),
        tuple(hessian_gradients),
    )
    if any(len(values) != len(names) for values in components):
        raise ValueError("gradient tuples must align with parameter_names")

    reference: Tensor | None = None
    for index, triple in enumerate(zip(*components, strict=True)):
        local = [value for value in triple if value is not None]
        for value in local:
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError("gradients must be floating tensors or null")
        if not local:
            continue
        first = local[0]
        for value in local[1:]:
            if value.shape != first.shape:
                raise ValueError(
                    f"gradient shapes are not aligned at parameter index {index}"
                )
            if value.device != first.device or value.dtype != first.dtype:
                raise ValueError(
                    "gradient device/dtype are not aligned at parameter "
                    f"index {index}"
                )
        if reference is None:
            reference = first
        elif first.device != reference.device or first.dtype != reference.dtype:
            raise ValueError("all gradient tensors must share device and dtype")
    if reference is None:
        raise ValueError("at least one aligned gradient tensor is required")
    return (
        names,
        components[0],
        components[1],
        components[2],
        groups,
        reference.device,
        reference.dtype,
    )


def _batched_finite_and_reachable_flags(
    components: tuple[GradientTuple, GradientTuple, GradientTuple],
    *,
    device: torch.device,
) -> tuple[tuple[tuple[bool, bool], ...], ...]:
    """Return ``(finite, reachable)`` flags using one host synchronization."""

    inactive = torch.tensor((True, False), dtype=torch.bool, device=device)
    rows: list[Tensor] = []
    for gradients in components:
        for gradient in gradients:
            if gradient is None:
                rows.append(inactive)
            else:
                rows.append(
                    torch.stack(
                        (
                            torch.isfinite(gradient.detach()).all(),
                            torch.any(gradient.detach() != 0.0),
                        )
                    )
                )
    # This is the adapter's only explicit device-to-host value transfer.  The
    # existing controller subsequently works on one tensor per active group.
    host_flags = torch.stack(rows).to(device="cpu").tolist()
    width = len(components[0])
    result: list[tuple[tuple[bool, bool], ...]] = []
    cursor = 0
    for _ in components:
        result.append(
            tuple(
                (bool(finite), bool(reachable))
                for finite, reachable in host_flags[cursor : cursor + width]
            )
        )
        cursor += width
    return tuple(result)


def _reference_for_index(
    index: int,
    energy: GradientTuple,
    force: GradientTuple,
    hessian: GradientTuple,
) -> Tensor:
    reference = next(
        (
            value
            for value in (energy[index], force[index], hessian[index])
            if value is not None
        ),
        None,
    )
    if reference is None:
        raise RuntimeError(
            "derivative-active parameter has no aligned tensor reference"
        )
    return reference


def _pack_component(
    gradients: GradientTuple,
    active_indices: Sequence[int],
    references: Mapping[int, Tensor],
) -> Tensor:
    pieces = [
        (
            torch.zeros_like(references[index])
            if gradients[index] is None
            else gradients[index].detach()
        ).reshape(-1)
        for index in active_indices
    ]
    return torch.cat(pieces, dim=0).detach()


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


def _full_gradient_metrics(
    combined: GradientTuple,
    force: GradientTuple,
    hessian: GradientTuple,
    energy: GradientTuple,
) -> dict[str, float]:
    """Compute complete-layout global metrics with one host transfer."""

    references = tuple(
        next(
            (
                value
                for value in (combined[index], energy[index], force[index], hessian[index])
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
        return torch.cat(pieces).to(dtype=torch.float64)

    combined_vector = flatten(combined)
    force_vector = flatten(force)
    hessian_vector = flatten(hessian)

    def stable_norm(vector: Tensor) -> Tensor:
        maximum = vector.abs().amax()
        safe_maximum = maximum.clamp_min(torch.finfo(torch.float64).tiny)
        return maximum * torch.sqrt(torch.square(vector / safe_maximum).sum())

    combined_norm = stable_norm(combined_vector)
    force_norm = stable_norm(force_vector)
    hessian_norm = stable_norm(hessian_vector)
    force_dot = torch.dot(combined_vector, force_vector)
    hessian_dot = torch.dot(combined_vector, hessian_vector)
    metrics = torch.stack(
        (combined_norm, force_norm, hessian_norm, force_dot, hessian_dot)
    ).to(device="cpu")
    (
        combined_norm_value,
        force_norm_value,
        hessian_norm_value,
        force_dot_value,
        hessian_dot_value,
    ) = (float(value) for value in metrics.tolist())
    return {
        "global_force_gradient_l2": force_norm_value,
        "global_hessian_gradient_l2": hessian_norm_value,
        "global_assigned_gradient_l2": combined_norm_value,
        "global_assigned_force_dot": force_dot_value,
        "global_assigned_hessian_dot": hessian_dot_value,
        "global_assigned_force_cosine": force_dot_value
        / (combined_norm_value * force_norm_value),
        "global_assigned_hessian_cosine": hessian_dot_value
        / (combined_norm_value * hessian_norm_value),
    }


def balance_blockwise_derivative_gradients_packed(
    parameter_names: Sequence[str],
    energy_gradients: Sequence[Tensor | None],
    force_gradients: Sequence[Tensor | None],
    hessian_gradients: Sequence[Tensor | None],
    *,
    parameter_groups: Mapping[str, str],
    derivative_target_ratio: float,
    force_target_ratio: float | None = None,
    hessian_target_ratio: float | None = None,
) -> BlockwiseBalanceResult:
    """Balance E/F/H gradients through one synthetic tensor per active group.

    The public contract mirrors
    :func:`balance_blockwise_derivative_gradients`.  Individual tensors for
    which both derivative tasks are null or exactly zero bypass packing and
    retain the detached energy gradient bit-for-bit.  The base controller is
    the sole implementation of normalization, projection, and task mixing.
    """

    (
        names,
        energy,
        force,
        hessian,
        grouped_indices,
        device,
        _dtype,
    ) = _as_aligned_gradient_tuples(
        parameter_names,
        energy_gradients,
        force_gradients,
        hessian_gradients,
        parameter_groups,
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

    packed_energy: list[Tensor] = []
    packed_force: list[Tensor] = []
    packed_hessian: list[Tensor] = []
    packed_names: list[str] = []
    packed_parameter_groups: dict[str, str] = {}
    packed_groups: list[_PackedGroup] = []

    for group, original_indices in grouped_indices.items():
        active_indices = tuple(
            index
            for index in original_indices
            if force_reachable[index] or hessian_reachable[index]
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
        packed_index = len(packed_names)
        synthetic_name = f"packed_group_{packed_index}"
        packed_names.append(synthetic_name)
        packed_parameter_groups[synthetic_name] = group
        packed_energy.append(_pack_component(energy, active_indices, references))
        packed_force.append(_pack_component(force, active_indices, references))
        packed_hessian.append(_pack_component(hessian, active_indices, references))
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

    if not packed_names:
        # Match the base controller's fail-closed behavior for an entirely
        # derivative-dead input without inventing a synthetic gradient.
        raise RuntimeError("zero global force gradient prevents balancing")

    packed_result = balance_blockwise_derivative_gradients(
        packed_names,
        packed_energy,
        packed_force,
        packed_hessian,
        parameter_groups=packed_parameter_groups,
        derivative_target_ratio=derivative_target_ratio,
        force_target_ratio=force_target_ratio,
        hessian_target_ratio=hessian_target_ratio,
    )

    output_fields: list[list[Tensor | None]] = [
        [None] * len(names) for _ in range(5)
    ]
    combined, projected, force_part, hessian_part, derivative_part = output_fields

    # Preserve derivative-dead energy values while removing any accidental
    # graph retained by a synthetic test input.
    all_active = {
        index for packed_group in packed_groups for index in packed_group.active_indices
    }
    for index in range(len(names)):
        if index not in all_active and energy[index] is not None:
            combined[index] = energy[index].detach()
            projected[index] = energy[index].detach()

    packed_fields = (
        packed_result.combined_gradients,
        packed_result.projected_energy_gradients,
        packed_result.force_contributions,
        packed_result.hessian_contributions,
        packed_result.derivative_contributions,
    )
    base_blocks = {
        block["group"]: block for block in packed_result.audit["blocks"]
    }
    for packed_group in packed_groups:
        for packed_field, output in zip(packed_fields, output_fields, strict=True):
            _split_component(
                packed_field[packed_group.packed_index], packed_group, output
            )
        # When no projection was required, the formal controller preserves a
        # null energy slot as null rather than manufacturing a zero tensor.
        projection = base_blocks[packed_group.name]["projection"]
        if not projection["projection_applied"]:
            for index in packed_group.active_indices:
                if energy[index] is None:
                    projected[index] = None

    group_audit: list[dict[str, Any]] = []
    packed_by_name = {group.name: group for group in packed_groups}
    for group, original_indices in grouped_indices.items():
        packed_group = packed_by_name.get(group)
        if packed_group is None:
            group_audit.append(
                {
                    "group": group,
                    "status": "derivative_dead",
                    "original_parameter_tensor_count": len(original_indices),
                    "derivative_active_tensor_count": 0,
                    "derivative_dead_tensor_count": len(original_indices),
                    "packed_scalar_count": 0,
                    "derivative_dead_energy_preserved": True,
                    "base_controller_block": None,
                }
            )
            continue
        group_audit.append(
            {
                "group": group,
                "status": "packed",
                "original_parameter_tensor_count": len(original_indices),
                "derivative_active_tensor_count": len(packed_group.active_indices),
                "derivative_dead_tensor_count": len(packed_group.dead_indices),
                "packed_scalar_count": sum(packed_group.scalar_counts),
                "derivative_dead_energy_preserved": all(
                    energy[index] is None
                    or torch.equal(combined[index], energy[index].detach())
                    for index in packed_group.dead_indices
                ),
                "base_controller_block": base_blocks[group],
            }
        )

    packed_audit = packed_result.audit
    global_metrics = _full_gradient_metrics(
        tuple(combined), force, hessian, energy
    )
    independent_task_ratios = force_target_ratio is not None
    audit: dict[str, Any] = {
        "schema_version": (
            INDEPENDENT_TASK_RATIO_SCHEMA_VERSION
            if independent_task_ratios
            else SCHEMA_VERSION
        ),
        "base_controller_schema_version": packed_audit["schema_version"],
        "derivative_target_ratio": packed_audit["derivative_target_ratio"],
        "original_parameter_tensor_count": len(names),
        "packed_parameter_tensor_count": len(packed_names),
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
        **global_metrics,
        "groups": group_audit,
        "base_controller_audit": packed_audit,
    }
    if independent_task_ratios:
        audit.update(
            {
                "force_target_ratio": packed_audit["force_target_ratio"],
                "hessian_target_ratio": packed_audit[
                    "hessian_target_ratio"
                ],
            }
        )
    return BlockwiseBalanceResult(
        combined_gradients=tuple(combined),
        projected_energy_gradients=tuple(projected),
        force_contributions=tuple(force_part),
        hessian_contributions=tuple(hessian_part),
        derivative_contributions=tuple(derivative_part),
        audit=audit,
    )


__all__ = [
    "INDEPENDENT_TASK_RATIO_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "balance_blockwise_derivative_gradients_packed",
]
