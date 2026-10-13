"""Stateless blockwise E/F/H gradient balancing for derivative recovery.

This module operates on already-computed, aligned parameter-gradient tuples.
It deliberately does not choose a parameter grouping or a target ratio for a
caller.  Version 1 uses one common derivative target ratio for force and
Hessian supervision.  In every block, the force and Hessian contributions are
constructed as

``rho * ||g_E||_reachable * unit(g_F)`` and
``rho * ||g_E||_reachable * unit(g_H)``.

When both tasks reach a block, their equal-norm sum has non-negative inner
product with both task gradients unless their unit directions are exactly
opposite.  The latter case fails closed.  The energy direction is projected
onto the intersection of the force and Hessian non-conflict half-spaces, then
renormalized to its original norm on the derivative-reachable subspace.
Derivative-dead parameter tensors retain their original energy gradient
exactly and never enter any balancing norm.

No large equivalent loss multiplier is materialized: tiny task gradients are
first converted to max-scaled unit directions and are only then multiplied by
the requested target amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


SCHEMA_VERSION = "equiso2-blockwise-derivative-balance-v1"
INDEPENDENT_TASK_RATIO_SCHEMA_VERSION = (
    "equiso2-blockwise-derivative-balance-independent-task-ratios-v1"
)
_BLOCK_PATTERN = re.compile(r"^blocks\.(\d+)(?:\.|$)")
GradientTuple = tuple[Tensor | None, ...]


@dataclass(frozen=True)
class BlockwiseBalanceResult:
    """Balanced gradients, explicit components, and JSON-safe diagnostics."""

    combined_gradients: GradientTuple
    projected_energy_gradients: GradientTuple
    force_contributions: GradientTuple
    hessian_contributions: GradientTuple
    derivative_contributions: GradientTuple
    audit: dict[str, Any]


def coarse_parameter_groups(parameter_names: Sequence[str]) -> dict[str, str]:
    """Return the registered coarse group for each supported parameter name.

    Supported groups are ``atom_embedding``, one group per ``blocks.<index>``,
    and ``energy_head``.  Unknown or duplicate names fail instead of being
    silently assigned to a catch-all group.  The returned mapping is intended
    to be passed explicitly to :func:`balance_blockwise_derivative_gradients`;
    the balancing API has no implicit grouping default.
    """

    result: dict[str, str] = {}
    for raw_name in parameter_names:
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("parameter names must be non-empty strings")
        if raw_name in result:
            raise ValueError(f"duplicate parameter name: {raw_name!r}")
        if raw_name == "atom_embedding" or raw_name.startswith(
            "atom_embedding."
        ):
            group = "atom_embedding"
        else:
            match = _BLOCK_PATTERN.match(raw_name)
            if match is not None:
                group = f"blocks.{int(match.group(1))}"
            elif raw_name == "energy_head" or raw_name.startswith(
                "energy_head."
            ):
                group = "energy_head"
            else:
                raise ValueError(
                    f"parameter {raw_name!r} is not covered by the coarse "
                    "blockwise grouping"
                )
        result[raw_name] = group
    if not result:
        raise ValueError("parameter_names cannot be empty")
    return result


def group_parameter_indices(
    parameter_names: Sequence[str],
    parameter_groups: Mapping[str, str],
) -> dict[str, tuple[int, ...]]:
    """Validate an explicit name-to-group mapping and return aligned indices."""

    names = tuple(parameter_names)
    if not names:
        raise ValueError("parameter_names cannot be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("parameter_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique")
    if set(parameter_groups) != set(names):
        missing = sorted(set(names).difference(parameter_groups))
        extra = sorted(set(parameter_groups).difference(names))
        raise ValueError(
            "parameter_groups must cover parameter_names exactly; "
            f"missing={missing}, extra={extra}"
        )
    grouped: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        group = parameter_groups[name]
        if not isinstance(group, str) or not group:
            raise ValueError(f"parameter group for {name!r} must be non-empty")
        grouped.setdefault(group, []).append(index)
    return {group: tuple(indices) for group, indices in grouped.items()}


def _finite_positive(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _as_gradient_tuple(
    values: Sequence[Tensor | None], *, name: str, expected: int
) -> GradientTuple:
    result = tuple(values)
    if len(result) != expected:
        raise ValueError(f"{name} gradients must align with parameter_names")
    for gradient in result:
        if gradient is None:
            continue
        if not isinstance(gradient, Tensor) or not gradient.is_floating_point():
            raise TypeError(f"{name} gradients must be floating tensors or null")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite {name} gradient")
    return result


def _validate_aligned_tensors(
    energy: GradientTuple,
    force: GradientTuple,
    hessian: GradientTuple,
) -> tuple[torch.device, torch.dtype]:
    reference: Tensor | None = None
    for index, triple in enumerate(zip(energy, force, hessian, strict=True)):
        local = [value for value in triple if value is not None]
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
                    f"gradient device/dtype are not aligned at index {index}"
                )
        if reference is None:
            reference = first
        elif first.device != reference.device or first.dtype != reference.dtype:
            raise ValueError("all gradient tensors must share device and dtype")
    if reference is None:
        raise ValueError("at least one aligned gradient tensor is required")
    return reference.device, reference.dtype


def _is_reachable(gradient: Tensor | None) -> bool:
    return gradient is not None and bool(torch.any(gradient != 0.0))


def _stable_norm(
    gradients: Sequence[Tensor | None], indices: Sequence[int]
) -> float:
    tensors = [
        gradients[index].detach()
        for index in indices
        if gradients[index] is not None
    ]
    if not tensors:
        return 0.0
    maximum = max(float(tensor.abs().amax().detach()) for tensor in tensors)
    if maximum == 0.0:
        return 0.0
    if not math.isfinite(maximum):
        raise FloatingPointError("non-finite maximum gradient magnitude")
    device = tensors[0].device
    squared = torch.zeros((), dtype=torch.float64, device=device)
    for tensor in tensors:
        scaled = tensor.to(dtype=torch.float64) / maximum
        squared = squared + scaled.square().sum(dtype=torch.float64)
    scaled_norm = math.sqrt(float(squared.detach()))
    norm = maximum * scaled_norm
    if not math.isfinite(norm):
        raise FloatingPointError("gradient norm is not representable in FP64")
    return norm


def _stable_dot(
    left: Sequence[Tensor | None],
    right: Sequence[Tensor | None],
    indices: Sequence[int],
) -> float:
    device = next(
        (
            value.device
            for index in indices
            for value in (left[index], right[index])
            if value is not None
        ),
        torch.device("cpu"),
    )
    result = torch.zeros((), dtype=torch.float64, device=device)
    for index in indices:
        left_value = left[index]
        right_value = right[index]
        if left_value is None or right_value is None:
            continue
        result = result + (
            left_value.detach().to(dtype=torch.float64)
            * right_value.detach().to(dtype=torch.float64)
        ).sum(dtype=torch.float64)
    value = float(result.detach())
    if not math.isfinite(value):
        raise FloatingPointError("gradient dot product is non-finite")
    return value


def _cosine(
    left: Sequence[Tensor | None],
    right: Sequence[Tensor | None],
    indices: Sequence[int],
) -> float | None:
    left_norm = _stable_norm(left, indices)
    right_norm = _stable_norm(right, indices)
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    value = _stable_dot(left, right, indices) / (left_norm * right_norm)
    if not math.isfinite(value):
        raise FloatingPointError("gradient cosine is non-finite")
    return max(-1.0, min(1.0, value))


def _zeros_for_indices(
    references: Sequence[Tensor | None], indices: Sequence[int]
) -> list[Tensor | None]:
    result: list[Tensor | None] = [None] * len(references)
    for index in indices:
        reference = references[index]
        if reference is not None:
            result[index] = torch.zeros_like(reference)
    return result


def _unit_direction(
    gradients: GradientTuple,
    indices: Sequence[int],
    references: GradientTuple,
    *,
    name: str,
) -> GradientTuple:
    active = [index for index in indices if _is_reachable(gradients[index])]
    norm = _stable_norm(gradients, active)
    if not active or norm == 0.0:
        raise RuntimeError(f"zero {name} gradient prevents unit construction")
    maximum = max(
        float(gradients[index].detach().abs().amax())  # type: ignore[union-attr]
        for index in active
    )
    device = gradients[active[0]].device  # type: ignore[union-attr]
    scaled_squared = torch.zeros((), dtype=torch.float64, device=device)
    for index in active:
        gradient = gradients[index]
        assert gradient is not None
        scaled = gradient.detach().to(dtype=torch.float64) / maximum
        scaled_squared = scaled_squared + scaled.square().sum(dtype=torch.float64)
    scaled_norm = math.sqrt(float(scaled_squared.detach()))
    if not math.isfinite(scaled_norm) or scaled_norm <= 0.0:
        raise FloatingPointError(f"invalid max-scaled {name} norm")
    result = _zeros_for_indices(references, indices)
    for index in active:
        gradient = gradients[index]
        assert gradient is not None
        result[index] = (gradient / maximum) / scaled_norm
    unit = tuple(result)
    realized = _stable_norm(unit, active)
    if not math.isfinite(realized) or realized <= 0.0:
        raise FloatingPointError(f"invalid {name} unit direction")
    # A second normalization corrects the small rounding error caused by
    # constructing the unit vector in the model's dtype.
    for index in active:
        assert result[index] is not None
        result[index] = result[index] / realized
    return tuple(result)


def _scaled_tuple(
    values: GradientTuple, indices: Sequence[int], scale: float
) -> GradientTuple:
    result: list[Tensor | None] = [None] * len(values)
    for index in indices:
        value = values[index]
        if value is not None:
            result[index] = value * value.new_tensor(scale)
    return tuple(result)


def _linear_combination(
    references: GradientTuple,
    indices: Sequence[int],
    terms: Sequence[tuple[float, GradientTuple]],
) -> GradientTuple:
    result: list[Tensor | None] = [None] * len(references)
    for index in indices:
        reference = references[index]
        if reference is None:
            reference = next(
                (values[index] for _, values in terms if values[index] is not None),
                None,
            )
        if reference is None:
            continue
        combined = torch.zeros_like(reference)
        for coefficient, values in terms:
            value = values[index]
            if value is not None:
                combined = combined + value * value.new_tensor(coefficient)
        result[index] = combined
    return tuple(result)


def _projection_tolerance(dtype: torch.dtype) -> float:
    return 256.0 * float(torch.finfo(dtype).eps)


def _post_projection_tolerance(
    dtype: torch.dtype, renormalization_factor: float
) -> float:
    """Bound projection residual plus rescale/re-dot roundoff.

    The first term propagates the residual admitted by the pre-rescale
    feasibility check.  The additional base tolerance covers the independent
    rounding incurred when the projected tensors are materialized at unit
    norm and their task dot products are reduced again.
    """

    tolerance = _projection_tolerance(dtype)
    return tolerance * (max(1.0, renormalization_factor) + 1.0)


def _project_energy_to_halfspaces(
    energy_unit: GradientTuple,
    force_unit: GradientTuple | None,
    hessian_unit: GradientTuple | None,
    indices: Sequence[int],
    *,
    dtype: torch.dtype,
) -> tuple[GradientTuple, dict[str, Any]]:
    """Project a unit energy direction onto one or two non-conflict halfspaces."""

    directions = [
        (name, value)
        for name, value in (("force", force_unit), ("hessian", hessian_unit))
        if value is not None
    ]
    if not directions:
        raise ValueError("energy projection requires a derivative direction")
    tolerance = _projection_tolerance(dtype)

    def feasible(candidate: GradientTuple) -> bool:
        return all(
            _stable_dot(candidate, direction, indices) >= -tolerance
            for _, direction in directions
        )

    candidates: list[tuple[str, GradientTuple, float]] = []
    if feasible(energy_unit):
        candidates.append(("none", energy_unit, 0.0))

    for name, direction in directions:
        denominator = _stable_dot(direction, direction, indices)
        inner = _stable_dot(energy_unit, direction, indices)
        multiplier = -inner / denominator
        if multiplier < 0.0:
            continue
        candidate = _linear_combination(
            energy_unit,
            indices,
            ((1.0, energy_unit), (multiplier, direction)),
        )
        if feasible(candidate):
            candidates.append((name, candidate, multiplier * multiplier * denominator))

    if len(directions) == 2:
        (_, first), (_, second) = directions
        ff = _stable_dot(first, first, indices)
        hh = _stable_dot(second, second, indices)
        fh = _stable_dot(first, second, indices)
        a = _stable_dot(energy_unit, first, indices)
        b = _stable_dot(energy_unit, second, indices)
        determinant = ff * hh - fh * fh
        if determinant > 0.0:
            first_multiplier = (-a * hh + b * fh) / determinant
            second_multiplier = (-b * ff + a * fh) / determinant
            if first_multiplier >= 0.0 and second_multiplier >= 0.0:
                candidate = _linear_combination(
                    energy_unit,
                    indices,
                    (
                        (1.0, energy_unit),
                        (first_multiplier, first),
                        (second_multiplier, second),
                    ),
                )
                if feasible(candidate):
                    distance_squared = (
                        first_multiplier * first_multiplier * ff
                        + second_multiplier * second_multiplier * hh
                        + 2.0
                        * first_multiplier
                        * second_multiplier
                        * fh
                    )
                    candidates.append(("force+hessian", candidate, distance_squared))

    if not candidates:
        # The closed-form two-constraint solve becomes ill-conditioned when
        # Force and Hessian directions are almost collinear.  The feasible
        # cone still contains the origin, so use deterministic cyclic
        # orthogonal projections as a numerical fallback.  Keeping this in
        # the controller dtype (CPU float64 in the distributed runner) avoids
        # introducing a host-side linear-algebra dependency.
        candidate = energy_unit
        for _ in range(256):
            changed = False
            for _, direction in directions:
                inner = _stable_dot(candidate, direction, indices)
                if inner < 0.0:
                    denominator = _stable_dot(direction, direction, indices)
                    candidate = _linear_combination(
                        candidate,
                        indices,
                        ((1.0, candidate), (-inner / denominator, direction)),
                    )
                    changed = True
            if feasible(candidate):
                delta = _linear_combination(
                    energy_unit,
                    indices,
                    ((1.0, candidate), (-1.0, energy_unit)),
                )
                distance_squared = _stable_dot(delta, delta, indices)
                candidates.append(
                    ("cyclic-numerical-fallback", candidate, distance_squared)
                )
                break
            if not changed:
                break
    if not candidates:
        zero = tuple(
            torch.zeros_like(value) if value is not None else None
            for value in energy_unit
        )
        return zero, {
            "active_set": "zero_fallback",
            "projection_applied": True,
            "zero_fallback": True,
            "projection_distance_l2": 1.0,
            "pre_renormalization_norm": 0.0,
            "renormalization_factor": 0.0,
            "post_unit_task_dots": {
                name: 0.0 for name, _ in directions
            },
            "numerical_tolerance": tolerance,
            "post_numerical_tolerance": tolerance,
        }
    # Stable sorting preserves the deterministic active-set enumeration for
    # exactly tied candidates.
    active_set, projected, distance_squared = min(
        candidates, key=lambda item: item[2]
    )
    projected_norm = _stable_norm(projected, indices)
    if projected_norm == 0.0:
        return projected, {
            "active_set": "zero_fallback",
            "projection_applied": True,
            "zero_fallback": True,
            "projection_distance_l2": math.sqrt(
                max(0.0, distance_squared)
            ),
            "pre_renormalization_norm": 0.0,
            "renormalization_factor": 0.0,
            "post_unit_task_dots": {
                name: 0.0 for name, _ in directions
            },
            "numerical_tolerance": tolerance,
            "post_numerical_tolerance": tolerance,
        }
    renormalization_factor = 1.0 / projected_norm
    renormalized = _scaled_tuple(projected, indices, renormalization_factor)
    post_dots = {
        name: _stable_dot(renormalized, direction, indices)
        for name, direction in directions
    }
    # ``feasible(projected)`` admits at most ``tolerance`` of absolute
    # roundoff at a halfspace boundary.  Positive renormalization scales that
    # residual, while materializing the rescaled tensors and independently
    # reducing the post-rescale dot product incur one additional base
    # tolerance.  This is only a post-projection numerical audit: the
    # subsequent block and global assigned-gradient cone checks still use the
    # original unscaled tolerance and remain the final safety invariant.
    post_tolerance = _post_projection_tolerance(
        dtype, renormalization_factor
    )
    if any(value < -post_tolerance for value in post_dots.values()):
        raise FloatingPointError(
            "renormalized energy violates a task halfspace: "
            f"active_set={active_set}, projected_norm={projected_norm:.17g}, "
            f"renormalization_factor={renormalization_factor:.17g}, "
            f"post_dots={post_dots}, post_tolerance={post_tolerance:.17g}"
        )
    return renormalized, {
        "active_set": active_set,
        "projection_applied": active_set != "none",
        "zero_fallback": False,
        "projection_distance_l2": math.sqrt(max(0.0, distance_squared)),
        "pre_renormalization_norm": projected_norm,
        "renormalization_factor": renormalization_factor,
        "post_unit_task_dots": post_dots,
        "numerical_tolerance": tolerance,
        "post_numerical_tolerance": post_tolerance,
    }


def balance_blockwise_derivative_gradients(
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
    """Construct a blockwise, conflict-safe E/F/H parameter gradient.

    ``parameter_groups`` and ``derivative_target_ratio`` are required keyword
    arguments.  By default that one legacy ratio is used for both derivative
    tasks.  Supplying both ``force_target_ratio`` and
    ``hessian_target_ratio`` activates the isolated independent-ratio mode;
    supplying only one is rejected.
    """

    names = tuple(parameter_names)
    groups = group_parameter_indices(names, parameter_groups)
    rho = _finite_positive(
        derivative_target_ratio, name="derivative_target_ratio"
    )
    independent_task_ratios = force_target_ratio is not None
    if independent_task_ratios != (hessian_target_ratio is not None):
        raise ValueError(
            "force_target_ratio and hessian_target_ratio must be supplied "
            "together"
        )
    rho_force = (
        rho
        if force_target_ratio is None
        else _finite_positive(
            force_target_ratio, name="force_target_ratio"
        )
    )
    rho_hessian = (
        rho
        if hessian_target_ratio is None
        else _finite_positive(
            hessian_target_ratio, name="hessian_target_ratio"
        )
    )
    energy = _as_gradient_tuple(energy_gradients, name="energy", expected=len(names))
    force = _as_gradient_tuple(force_gradients, name="force", expected=len(names))
    hessian = _as_gradient_tuple(
        hessian_gradients, name="hessian", expected=len(names)
    )
    _, dtype = _validate_aligned_tensors(energy, force, hessian)
    all_indices = tuple(range(len(names)))
    global_force_norm = _stable_norm(force, all_indices)
    global_hessian_norm = _stable_norm(hessian, all_indices)
    if global_force_norm == 0.0:
        raise RuntimeError("zero global force gradient prevents balancing")
    if global_hessian_norm == 0.0:
        raise RuntimeError("zero global Hessian gradient prevents balancing")

    combined: list[Tensor | None] = [None] * len(names)
    projected_energy: list[Tensor | None] = [None] * len(names)
    force_contribution: list[Tensor | None] = [None] * len(names)
    hessian_contribution: list[Tensor | None] = [None] * len(names)
    derivative_contribution: list[Tensor | None] = [None] * len(names)
    block_audits: list[dict[str, Any]] = []

    for group, indices in groups.items():
        force_active = tuple(index for index in indices if _is_reachable(force[index]))
        hessian_active = tuple(
            index for index in indices if _is_reachable(hessian[index])
        )
        derivative_active = tuple(sorted(set(force_active) | set(hessian_active)))
        derivative_dead = tuple(
            index for index in indices if index not in set(derivative_active)
        )
        for index in derivative_dead:
            combined[index] = energy[index]
            projected_energy[index] = energy[index]
        if not derivative_active:
            block_audits.append(
                {
                    "group": group,
                    "parameter_tensor_count": len(indices),
                    "force_reachable_tensor_count": 0,
                    "hessian_reachable_tensor_count": 0,
                    "derivative_reachable_tensor_count": 0,
                    "derivative_dead_tensor_count": len(indices),
                    "derivative_dead_energy_preserved": True,
                    "status": "derivative_dead",
                }
            )
            continue

        energy_reachable_norm = _stable_norm(energy, derivative_active)
        if energy_reachable_norm == 0.0:
            raise RuntimeError(
                f"zero energy gradient on derivative-reachable block {group!r}"
            )
        references_list: list[Tensor | None] = [None] * len(names)
        for index in derivative_active:
            references_list[index] = next(
                value
                for value in (energy[index], force[index], hessian[index])
                if value is not None
            )
        references = tuple(references_list)
        energy_unit = _unit_direction(
            energy,
            derivative_active,
            references,
            name=f"energy/{group}",
        )
        force_unit = (
            _unit_direction(
                force,
                derivative_active,
                references,
                name=f"force/{group}",
            )
            if force_active
            else None
        )
        hessian_unit = (
            _unit_direction(
                hessian,
                derivative_active,
                references,
                name=f"Hessian/{group}",
            )
            if hessian_active
            else None
        )
        safe_energy_unit, projection_audit = _project_energy_to_halfspaces(
            energy_unit,
            force_unit,
            hessian_unit,
            derivative_active,
            dtype=dtype,
        )
        if projection_audit["projection_applied"]:
            safe_energy = _scaled_tuple(
                safe_energy_unit, derivative_active, energy_reachable_norm
            )
        else:
            # A feasible energy direction is not merely directionally
            # equivalent: preserve its original tensors bit-for-bit.
            safe_energy_values: list[Tensor | None] = [None] * len(names)
            for index in derivative_active:
                safe_energy_values[index] = energy[index]
            safe_energy = tuple(safe_energy_values)
        force_target_amplitude = rho_force * energy_reachable_norm
        hessian_target_amplitude = rho_hessian * energy_reachable_norm
        force_part = (
            _scaled_tuple(
                force_unit, derivative_active, force_target_amplitude
            )
            if force_unit is not None
            else tuple([None] * len(names))
        )
        hessian_part = (
            _scaled_tuple(
                hessian_unit, derivative_active, hessian_target_amplitude
            )
            if hessian_unit is not None
            else tuple([None] * len(names))
        )
        derivative_part = _linear_combination(
            references,
            derivative_active,
            ((1.0, force_part), (1.0, hessian_part)),
        )
        derivative_norm = _stable_norm(derivative_part, derivative_active)
        if derivative_norm == 0.0:
            raise RuntimeError(
                f"force and Hessian directions are exactly antiparallel in {group!r}"
            )
        block_combined = _linear_combination(
            references,
            derivative_active,
            ((1.0, safe_energy), (1.0, derivative_part)),
        )
        tolerance = _projection_tolerance(dtype)
        assigned_norm = _stable_norm(block_combined, derivative_active)
        task_dots: dict[str, float | None] = {"force": None, "hessian": None}
        task_cosines: dict[str, float | None] = {"force": None, "hessian": None}
        for task_name, raw, raw_norm in (
            ("force", force, _stable_norm(force, indices)),
            ("hessian", hessian, _stable_norm(hessian, indices)),
        ):
            if raw_norm == 0.0:
                continue
            dot = _stable_dot(block_combined, raw, indices)
            threshold = -tolerance * assigned_norm * raw_norm
            if dot < threshold:
                raise FloatingPointError(
                    f"assigned block gradient conflicts with {task_name} in {group!r}"
                )
            task_dots[task_name] = dot
            task_cosines[task_name] = dot / (assigned_norm * raw_norm)

        for index in derivative_active:
            combined[index] = block_combined[index]
            projected_energy[index] = safe_energy[index]
            force_contribution[index] = force_part[index]
            hessian_contribution[index] = hessian_part[index]
            derivative_contribution[index] = derivative_part[index]

        before_full_energy_norm = _stable_norm(energy, indices)
        after_energy_values = list(safe_energy)
        for index in derivative_dead:
            after_energy_values[index] = energy[index]
        after_full_energy_norm = _stable_norm(after_energy_values, indices)
        norm_tolerance = tolerance * max(1.0, before_full_energy_norm)
        if (
            not projection_audit["zero_fallback"]
            and abs(after_full_energy_norm - before_full_energy_norm)
            > norm_tolerance
        ):
            raise FloatingPointError("energy block norm was not preserved")
        f_h_cosine = (
            _stable_dot(force_unit, hessian_unit, derivative_active)
            if force_unit is not None and hessian_unit is not None
            else None
        )
        block_audit = {
                "group": group,
                "status": "balanced",
                "parameter_tensor_count": len(indices),
                "force_reachable_tensor_count": len(force_active),
                "hessian_reachable_tensor_count": len(hessian_active),
                "derivative_reachable_tensor_count": len(derivative_active),
                "derivative_dead_tensor_count": len(derivative_dead),
                "derivative_dead_energy_preserved": all(
                    combined[index] is energy[index] for index in derivative_dead
                ),
                "energy_gradient_l2_before": before_full_energy_norm,
                "energy_gradient_l2_after_projection": after_full_energy_norm,
                "energy_reachable_gradient_l2": energy_reachable_norm,
                "force_gradient_l2": _stable_norm(force, indices),
                "hessian_gradient_l2": _stable_norm(hessian, indices),
                "derivative_target_ratio": rho,
                "target_task_contribution_l2": (
                    rho * energy_reachable_norm
                ),
                "force_contribution_l2": (
                    _stable_norm(force_part, derivative_active)
                    if force_unit is not None
                    else None
                ),
                "hessian_contribution_l2": (
                    _stable_norm(hessian_part, derivative_active)
                    if hessian_unit is not None
                    else None
                ),
                "derivative_contribution_l2": derivative_norm,
                "force_hessian_unit_cosine": f_h_cosine,
                "energy_force_cosine_before": (
                    _cosine(energy, force, derivative_active)
                    if force_unit is not None
                    else None
                ),
                "energy_hessian_cosine_before": (
                    _cosine(energy, hessian, derivative_active)
                    if hessian_unit is not None
                    else None
                ),
                "projected_energy_force_cosine": (
                    _cosine(safe_energy, force, derivative_active)
                    if force_unit is not None
                    else None
                ),
                "projected_energy_hessian_cosine": (
                    _cosine(safe_energy, hessian, derivative_active)
                    if hessian_unit is not None
                    else None
                ),
                "assigned_force_dot": task_dots["force"],
                "assigned_hessian_dot": task_dots["hessian"],
                "assigned_force_cosine": task_cosines["force"],
                "assigned_hessian_cosine": task_cosines["hessian"],
                "assigned_gradient_l2": assigned_norm,
                "projection": projection_audit,
            }
        if independent_task_ratios:
            block_audit.update(
                {
                    "force_target_ratio": rho_force,
                    "hessian_target_ratio": rho_hessian,
                    "target_task_contribution_l2": (
                        force_target_amplitude
                        if rho_force == rho_hessian
                        else None
                    ),
                    "target_force_contribution_l2": (
                        force_target_amplitude
                    ),
                    "target_hessian_contribution_l2": (
                        hessian_target_amplitude
                    ),
                }
            )
        block_audits.append(block_audit)

    combined_tuple = tuple(combined)
    global_combined_norm = _stable_norm(combined_tuple, all_indices)
    tolerance = _projection_tolerance(dtype)
    global_force_dot = _stable_dot(combined_tuple, force, all_indices)
    global_hessian_dot = _stable_dot(combined_tuple, hessian, all_indices)
    if global_force_dot < -tolerance * global_combined_norm * global_force_norm:
        raise FloatingPointError("assigned global gradient conflicts with force")
    if (
        global_hessian_dot
        < -tolerance * global_combined_norm * global_hessian_norm
    ):
        raise FloatingPointError("assigned global gradient conflicts with Hessian")
    audit = {
        "schema_version": (
            INDEPENDENT_TASK_RATIO_SCHEMA_VERSION
            if independent_task_ratios
            else SCHEMA_VERSION
        ),
        "derivative_target_ratio": rho,
        "parameter_tensor_count": len(names),
        "group_count": len(groups),
        "global_force_gradient_l2": global_force_norm,
        "global_hessian_gradient_l2": global_hessian_norm,
        "global_assigned_gradient_l2": global_combined_norm,
        "global_assigned_force_dot": global_force_dot,
        "global_assigned_hessian_dot": global_hessian_dot,
        "global_assigned_force_cosine": global_force_dot
        / (global_combined_norm * global_force_norm),
        "global_assigned_hessian_cosine": global_hessian_dot
        / (global_combined_norm * global_hessian_norm),
        "numerical_tolerance": tolerance,
        "blocks": block_audits,
    }
    if independent_task_ratios:
        audit.update(
            {
                "force_target_ratio": rho_force,
                "hessian_target_ratio": rho_hessian,
            }
        )
    return BlockwiseBalanceResult(
        combined_gradients=combined_tuple,
        projected_energy_gradients=tuple(projected_energy),
        force_contributions=tuple(force_contribution),
        hessian_contributions=tuple(hessian_contribution),
        derivative_contributions=tuple(derivative_contribution),
        audit=audit,
    )


def assign_parameter_gradients(
    parameters: Sequence[Tensor], gradients: Sequence[Tensor | None]
) -> dict[str, Any]:
    """Assign an aligned detached gradient tuple to trainable parameters."""

    parameter_tuple = tuple(parameters)
    gradient_tuple = tuple(gradients)
    if not parameter_tuple or len(parameter_tuple) != len(gradient_tuple):
        raise ValueError("parameters and gradients must be non-empty and aligned")
    active = 0
    # Validate the whole tuple before writing any ``.grad`` buffer.  An error
    # at a later parameter therefore cannot leave a partially assigned step.
    for index, (parameter, gradient) in enumerate(
        zip(parameter_tuple, gradient_tuple, strict=True)
    ):
        if not isinstance(parameter, Tensor) or not parameter.requires_grad:
            raise ValueError("all parameters must be trainable tensors")
        if gradient is None:
            continue
        if (
            gradient.shape != parameter.shape
            or gradient.device != parameter.device
            or gradient.dtype != parameter.dtype
        ):
            raise ValueError(f"gradient does not match parameter index {index}")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("cannot assign a non-finite gradient")
        active += int(bool(torch.any(gradient != 0.0)))
    if active == 0:
        raise RuntimeError("zero assigned gradient prevents an optimizer update")
    for parameter, gradient in zip(parameter_tuple, gradient_tuple, strict=True):
        parameter.grad = None if gradient is None else gradient.detach().clone()
    return {
        "active_parameter_tensor_count": active,
        "assigned_gradient_l2": _stable_norm(
            gradient_tuple, tuple(range(len(gradient_tuple)))
        ),
    }


def audit_realized_parameter_delta(
    parameters_after: Sequence[Tensor],
    parameters_before: Sequence[Tensor],
    force_gradients: Sequence[Tensor | None],
    hessian_gradients: Sequence[Tensor | None],
) -> dict[str, Any]:
    """Audit ``<g_F, delta>`` and ``<g_H, delta>`` for a realized update.

    ``delta`` is ``parameter_after - parameter_before``.  A first-order
    improving optimizer update therefore has a negative task dot product.
    This helper intentionally observes the actual optimizer result rather than
    inferring it from the gradient assigned before AdamW preconditioning.
    """

    after = tuple(parameters_after)
    before = tuple(parameters_before)
    if not after or len(after) != len(before):
        raise ValueError("before/after parameter tuples must be non-empty and aligned")
    force = _as_gradient_tuple(force_gradients, name="force", expected=len(after))
    hessian = _as_gradient_tuple(
        hessian_gradients, name="hessian", expected=len(after)
    )
    deltas: list[Tensor | None] = []
    for index, (current, previous) in enumerate(zip(after, before, strict=True)):
        if not isinstance(current, Tensor) or not isinstance(previous, Tensor):
            raise TypeError("before/after parameters must be tensors")
        if (
            current.shape != previous.shape
            or current.device != previous.device
            or current.dtype != previous.dtype
        ):
            raise ValueError(f"before/after mismatch at parameter index {index}")
        delta = current.detach() - previous.detach()
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("realized parameter delta is non-finite")
        for task_gradient in (force[index], hessian[index]):
            if task_gradient is not None and (
                task_gradient.shape != current.shape
                or task_gradient.device != current.device
                or task_gradient.dtype != current.dtype
            ):
                raise ValueError(
                    f"task gradient does not match parameter index {index}"
                )
        deltas.append(delta)
    delta_tuple = tuple(deltas)
    indices = tuple(range(len(after)))
    delta_norm = _stable_norm(delta_tuple, indices)
    force_norm = _stable_norm(force, indices)
    hessian_norm = _stable_norm(hessian, indices)
    force_dot = _stable_dot(force, delta_tuple, indices)
    hessian_dot = _stable_dot(hessian, delta_tuple, indices)
    return {
        "parameter_delta_l2": delta_norm,
        "force_gradient_l2": force_norm,
        "hessian_gradient_l2": hessian_norm,
        "force_dot_delta": force_dot,
        "hessian_dot_delta": hessian_dot,
        "force_first_order_improving": force_dot < 0.0,
        "hessian_first_order_improving": hessian_dot < 0.0,
        "force_delta_cosine": (
            None
            if force_norm == 0.0 or delta_norm == 0.0
            else force_dot / (force_norm * delta_norm)
        ),
        "hessian_delta_cosine": (
            None
            if hessian_norm == 0.0 or delta_norm == 0.0
            else hessian_dot / (hessian_norm * delta_norm)
        ),
    }


__all__ = [
    "BlockwiseBalanceResult",
    "INDEPENDENT_TASK_RATIO_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "assign_parameter_gradients",
    "audit_realized_parameter_delta",
    "balance_blockwise_derivative_gradients",
    "coarse_parameter_groups",
    "group_parameter_indices",
]
