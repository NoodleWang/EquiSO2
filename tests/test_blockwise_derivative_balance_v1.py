from __future__ import annotations

import math

import pytest
import torch

from equiso2.train.blockwise_derivative_balance_v1 import (
    assign_parameter_gradients,
    audit_realized_parameter_delta,
    balance_blockwise_derivative_gradients,
    coarse_parameter_groups,
    group_parameter_indices,
)


DTYPE = torch.float32


def _balance(
    energy,
    force,
    hessian,
    *,
    names=None,
    groups=None,
    ratio=1.0,
):
    if names is None:
        names = tuple(f"blocks.0.tensor_{index}" for index in range(len(energy)))
    if groups is None:
        groups = {name: "blocks.0" for name in names}
    return balance_blockwise_derivative_gradients(
        names,
        energy,
        force,
        hessian,
        parameter_groups=groups,
        derivative_target_ratio=ratio,
    )


def test_coarse_grouping_is_unique_complete_and_block_indexed() -> None:
    names = (
        "atom_embedding.weight",
        "blocks.0.message_scale",
        "blocks.0.dynamic.kernel_heads.0.0.weight",
        "blocks.12.gate.scalar_map.weight",
        "energy_head.network.0.weight",
        "energy_head.network.4.bias",
    )
    groups = coarse_parameter_groups(names)
    assert groups == {
        "atom_embedding.weight": "atom_embedding",
        "blocks.0.message_scale": "blocks.0",
        "blocks.0.dynamic.kernel_heads.0.0.weight": "blocks.0",
        "blocks.12.gate.scalar_map.weight": "blocks.12",
        "energy_head.network.0.weight": "energy_head",
        "energy_head.network.4.bias": "energy_head",
    }
    indices = group_parameter_indices(names, groups)
    assert indices == {
        "atom_embedding": (0,),
        "blocks.0": (1, 2),
        "blocks.12": (3,),
        "energy_head": (4, 5),
    }
    assert sorted(index for block in indices.values() for index in block) == list(
        range(len(names))
    )


@pytest.mark.parametrize(
    "names",
    [
        ("blocks.0.weight", "blocks.0.weight"),
        ("direct_head.weight",),
        ("",),
    ],
)
def test_coarse_grouping_rejects_duplicate_unknown_or_empty_names(names) -> None:
    with pytest.raises(ValueError):
        coarse_parameter_groups(names)


def test_energy_bias_is_derivative_dead_and_excluded_from_target_norm() -> None:
    names = ("energy_head.network.0.weight", "energy_head.network.4.bias")
    groups = coarse_parameter_groups(names)
    energy = (
        torch.tensor([-1.0, 1.0, 1.0], dtype=DTYPE),
        torch.tensor([100.0], dtype=DTYPE),
    )
    force = (
        torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        None,
    )
    hessian = (
        torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
        torch.zeros(1, dtype=DTYPE),
    )
    result = _balance(
        energy,
        force,
        hessian,
        names=names,
        groups=groups,
        ratio=2.0,
    )
    block = result.audit["blocks"][0]
    assert block["energy_reachable_gradient_l2"] == pytest.approx(math.sqrt(3.0))
    assert block["target_task_contribution_l2"] == pytest.approx(
        2.0 * math.sqrt(3.0)
    )
    assert result.combined_gradients[1] is energy[1]
    assert torch.equal(result.combined_gradients[1], energy[1])
    assert block["derivative_dead_tensor_count"] == 1
    assert block["derivative_dead_energy_preserved"] is True


def test_tiny_float32_task_scale_does_not_change_balanced_gradient() -> None:
    energy = (torch.tensor([0.5, 1.0, -0.25], dtype=DTYPE),)
    force = (torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),)
    ordinary = _balance(energy, force, hessian, ratio=1.25)
    tiny = _balance(
        energy,
        (force[0] * 1.0e-30,),
        (hessian[0] * 1.0e-30,),
        ratio=1.25,
    )
    assert torch.allclose(
        ordinary.combined_gradients[0],
        tiny.combined_gradients[0],
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert torch.allclose(
        ordinary.force_contributions[0],
        tiny.force_contributions[0],
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert torch.allclose(
        ordinary.hessian_contributions[0],
        tiny.hessian_contributions[0],
        rtol=2.0e-6,
        atol=2.0e-7,
    )


def test_negative_energy_conflict_is_projected_to_both_halfspaces() -> None:
    energy = (torch.tensor([-1.0, 1.0, 1.0], dtype=DTYPE),)
    force = (torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),)
    result = _balance(energy, force, hessian)
    block = result.audit["blocks"][0]
    projected = result.projected_energy_gradients[0]
    assert projected is not None
    assert block["projection"]["projection_applied"] is True
    assert float(torch.dot(projected, force[0])) >= -1.0e-6
    assert float(torch.dot(projected, hessian[0])) >= -1.0e-6
    assert torch.linalg.vector_norm(projected).item() == pytest.approx(
        torch.linalg.vector_norm(energy[0]).item(), rel=2.0e-6
    )
    assert block["assigned_force_dot"] >= -1.0e-6
    assert block["assigned_hessian_dot"] >= -1.0e-6
    assert result.audit["global_assigned_force_dot"] >= -1.0e-6
    assert result.audit["global_assigned_hessian_dot"] >= -1.0e-6


def test_incompatible_opposite_halfspaces_use_audited_zero_fallback() -> None:
    from equiso2.train.blockwise_derivative_balance_v1 import (
        _project_energy_to_halfspaces,
    )

    energy = (torch.tensor([1.0, 0.0], dtype=DTYPE),)
    force = (torch.tensor([1.0, 0.0], dtype=DTYPE),)
    hessian = (torch.tensor([-1.0, 0.0], dtype=DTYPE),)

    projected, audit = _project_energy_to_halfspaces(
        energy, force, hessian, (0,), dtype=DTYPE
    )

    assert projected[0] is not None
    assert torch.equal(projected[0], torch.zeros_like(projected[0]))
    assert audit["active_set"] == "zero_fallback"
    assert audit["zero_fallback"] is True


def test_projection_propagates_boundary_tolerance_through_renormalization() -> None:
    """A positive rescale must also rescale admitted boundary roundoff."""

    from equiso2.train.blockwise_derivative_balance_v1 import (
        _project_energy_to_halfspaces,
    )

    a = 0.6677378
    b = math.sqrt(1.0 - a * a)
    h_y = -2.71e-5 / b
    energy = (torch.tensor([-a, b, 0.0], dtype=torch.float32),)
    force = (torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),)
    hessian = (
        torch.tensor(
            [math.sqrt(1.0 - h_y * h_y), h_y, 0.0],
            dtype=torch.float32,
        ),
    )

    projected, audit = _project_energy_to_halfspaces(
        energy,
        force,
        hessian,
        (0,),
        dtype=torch.float32,
    )

    assert audit["active_set"] == "force"
    assert audit["renormalization_factor"] > 1.0
    assert audit["post_unit_task_dots"]["hessian"] < -audit["numerical_tolerance"]
    assert (
        audit["post_unit_task_dots"]["hessian"]
        >= -audit["post_numerical_tolerance"]
    )
    assert torch.isfinite(projected[0]).all()


def test_post_projection_tolerance_includes_independent_redot_roundoff() -> None:
    """Regression for the harmless FP64 boundary residual from Job 145."""

    from equiso2.train.blockwise_derivative_balance_v1 import (
        _post_projection_tolerance,
        _projection_tolerance,
    )

    renormalization_factor = 1.4867693119130905
    observed_boundary_dot = -8.451529406872105e-14
    base_tolerance = _projection_tolerance(torch.float64)
    propagated_only = base_tolerance * renormalization_factor
    post_tolerance = _post_projection_tolerance(
        torch.float64, renormalization_factor
    )

    # The old check rejected this dot by only 2.24e-18.  One independent
    # base-tolerance budget for materialization/reduction accepts it without
    # changing either the projected direction or the final cone checks.
    assert observed_boundary_dot < -propagated_only
    assert observed_boundary_dot >= -post_tolerance
    assert post_tolerance == pytest.approx(
        propagated_only + base_tolerance, rel=0.0, abs=1.0e-28
    )
    # The extra roundoff budget remains microscopic and cannot hide a real
    # halfspace violation.
    assert -1.0e-10 < -post_tolerance


def test_nonconflicting_energy_direction_is_not_projected() -> None:
    energy = (torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE),)
    force = (torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),)
    result = _balance(energy, force, hessian)
    block = result.audit["blocks"][0]
    assert block["projection"]["active_set"] == "none"
    assert block["projection"]["projection_applied"] is False
    assert result.projected_energy_gradients[0] is energy[0]
    assert torch.equal(result.projected_energy_gradients[0], energy[0])


def test_each_task_contribution_hits_the_same_explicit_block_target_norm() -> None:
    energy = (torch.tensor([3.0, 4.0, 0.0], dtype=DTYPE),)
    force = (torch.tensor([0.0, 2.0, 0.0], dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 0.0, -7.0], dtype=DTYPE),)
    result = _balance(energy, force, hessian, ratio=0.75)
    expected = 0.75 * 5.0
    block = result.audit["blocks"][0]
    assert torch.linalg.vector_norm(result.force_contributions[0]).item() == (
        pytest.approx(expected, rel=2.0e-6)
    )
    assert torch.linalg.vector_norm(result.hessian_contributions[0]).item() == (
        pytest.approx(expected, rel=2.0e-6)
    )
    assert block["force_contribution_l2"] == pytest.approx(expected, rel=2.0e-6)
    assert block["hessian_contribution_l2"] == pytest.approx(
        expected, rel=2.0e-6
    )


def test_wholly_derivative_dead_block_preserves_energy() -> None:
    names = ("blocks.0.weight", "energy_head.network.4.bias")
    groups = {
        "blocks.0.weight": "blocks.0",
        "energy_head.network.4.bias": "energy_head",
    }
    energy = (
        torch.tensor([1.0, 2.0], dtype=DTYPE),
        torch.tensor([9.0], dtype=DTYPE),
    )
    force = (torch.tensor([1.0, 0.0], dtype=DTYPE), None)
    hessian = (torch.tensor([0.0, 1.0], dtype=DTYPE), None)
    result = _balance(
        energy, force, hessian, names=names, groups=groups, ratio=1.0
    )
    dead = next(
        block
        for block in result.audit["blocks"]
        if block["group"] == "energy_head"
    )
    assert dead["status"] == "derivative_dead"
    assert result.combined_gradients[1] is energy[1]
    assert torch.equal(result.combined_gradients[1], energy[1])


def test_exactly_antiparallel_force_hessian_directions_fail_closed() -> None:
    energy = (torch.tensor([0.0, 1.0], dtype=DTYPE),)
    force = (torch.tensor([1.0, 0.0], dtype=DTYPE),)
    hessian = (torch.tensor([-1.0, 0.0], dtype=DTYPE),)
    with pytest.raises(RuntimeError, match="antiparallel"):
        _balance(energy, force, hessian)


@pytest.mark.parametrize("zero_task", ["force", "hessian"])
def test_zero_global_derivative_task_fails_closed(zero_task: str) -> None:
    energy = (torch.tensor([1.0, 1.0], dtype=DTYPE),)
    force = (torch.tensor([1.0, 0.0], dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 1.0], dtype=DTYPE),)
    if zero_task == "force":
        force = (torch.zeros(2, dtype=DTYPE),)
    else:
        hessian = (None,)
    with pytest.raises(RuntimeError, match=f"global {zero_task if zero_task == 'force' else 'Hessian'}"):
        _balance(energy, force, hessian)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_component_fails_closed(bad: float) -> None:
    energy = (torch.tensor([1.0, 1.0], dtype=DTYPE),)
    force = (torch.tensor([bad, 0.0], dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 1.0], dtype=DTYPE),)
    with pytest.raises(FloatingPointError, match="non-finite force"):
        _balance(energy, force, hessian)


def test_missing_explicit_group_and_nonpositive_ratio_fail_closed() -> None:
    names = ("blocks.0.weight",)
    values = (torch.ones(2, dtype=DTYPE),)
    with pytest.raises(ValueError, match="cover parameter_names exactly"):
        _balance(values, values, values, names=names, groups={})
    with pytest.raises(ValueError, match="finite and positive"):
        _balance(values, values, values, names=names, ratio=0.0)


def test_assign_parameter_gradients_is_exact_and_aligned() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=DTYPE))
    second = torch.nn.Parameter(torch.tensor([-1.0], dtype=DTYPE))
    gradients = (
        torch.tensor([3.0, -4.0], dtype=DTYPE),
        torch.tensor([5.0], dtype=DTYPE),
    )
    audit = assign_parameter_gradients((first, second), gradients)
    assert torch.equal(first.grad, gradients[0])
    assert torch.equal(second.grad, gradients[1])
    assert first.grad is not gradients[0]
    assert audit["active_parameter_tensor_count"] == 2
    assert audit["assigned_gradient_l2"] == pytest.approx(math.sqrt(50.0))


def test_assignment_validation_is_atomic() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0], dtype=DTYPE))
    second = torch.nn.Parameter(torch.tensor([2.0], dtype=DTYPE))
    first.grad = torch.tensor([9.0], dtype=DTYPE)
    second.grad = torch.tensor([8.0], dtype=DTYPE)
    with pytest.raises(ValueError, match="parameter index 1"):
        assign_parameter_gradients(
            (first, second),
            (
                torch.tensor([3.0], dtype=DTYPE),
                torch.tensor([4.0, 5.0], dtype=DTYPE),
            ),
        )
    assert torch.equal(first.grad, torch.tensor([9.0], dtype=DTYPE))
    assert torch.equal(second.grad, torch.tensor([8.0], dtype=DTYPE))


def test_realized_adamw_delta_audits_both_task_directional_derivatives() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=DTYPE))
    second = torch.nn.Parameter(torch.tensor([0.5], dtype=DTYPE))
    parameters = (first, second)
    force = (
        torch.tensor([1.0, 0.0], dtype=DTYPE),
        torch.tensor([0.0], dtype=DTYPE),
    )
    hessian = (
        torch.tensor([0.0, -2.0], dtype=DTYPE),
        torch.tensor([3.0], dtype=DTYPE),
    )
    assigned = (
        force[0] + hessian[0],
        force[1] + hessian[1],
    )
    before = tuple(parameter.detach().clone() for parameter in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-2, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    assign_parameter_gradients(parameters, assigned)
    optimizer.step()
    audit = audit_realized_parameter_delta(parameters, before, force, hessian)
    assert audit["parameter_delta_l2"] > 0.0
    assert audit["force_dot_delta"] < 0.0
    assert audit["hessian_dot_delta"] < 0.0
    assert audit["force_first_order_improving"] is True
    assert audit["hessian_first_order_improving"] is True
