from __future__ import annotations

import math

import pytest
import torch

from equiso2.train.blockwise_sobolev_balance_packed_v1 import (
    SCHEMA_VERSION,
    balance_blockwise_sobolev_gradients_packed,
)


DTYPE = torch.float64


def _balance(
    energy: tuple[torch.Tensor | None, ...],
    force: tuple[torch.Tensor | None, ...],
    hessian: tuple[torch.Tensor | None, ...],
    *,
    names: tuple[str, ...] | None = None,
    groups: dict[str, str] | None = None,
    force_ratio: float = 0.0,
    hessian_ratio: float = 1.0,
):
    if names is None:
        names = tuple(f"blocks.0.parameter_{index}" for index in range(len(energy)))
    if groups is None:
        groups = {name: "blocks.0" for name in names}
    return balance_blockwise_sobolev_gradients_packed(
        names,
        energy,
        force,
        hessian,
        parameter_groups=groups,
        force_target_ratio=force_ratio,
        hessian_target_ratio=hessian_ratio,
    )


def test_strict_eh_has_no_force_contribution_or_force_gate() -> None:
    energy = (torch.tensor([1.0, 1.0], dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 1.0], dtype=DTYPE),)
    result = _balance(energy, (None,), hessian)

    expected_hessian = torch.tensor([0.0, math.sqrt(2.0)], dtype=DTYPE)
    torch.testing.assert_close(result.hessian_contributions[0], expected_hessian)
    torch.testing.assert_close(
        result.combined_gradients[0], energy[0] + expected_hessian
    )
    assert result.force_contributions == (None,)
    assert result.audit["schema_version"] == SCHEMA_VERSION
    assert result.audit["controller_semantics"] == (
        "hard_hessian_soft_force_anchor_v1"
    )
    assert result.audit["strict_eh"] is True
    assert result.audit["force_enabled"] is False
    assert result.audit["force_is_hard_gate"] is False
    assert result.audit["hessian_is_hard_gate"] is True
    assert result.audit["global_force_gradient_l2"] is None
    assert result.audit["global_assigned_force_dot"] is None
    assert result.audit["global_soft_force_compatible"] is None
    assert result.audit["global_hessian_nonconflict_passed"] is True


@pytest.mark.parametrize(
    "force",
    [
        (torch.zeros(2, dtype=DTYPE),),
        (torch.tensor([1.0, 0.0], dtype=DTYPE),),
    ],
)
def test_strict_eh_rejects_even_zero_force_gradient_tensors(
    force: tuple[torch.Tensor, ...],
) -> None:
    with pytest.raises(ValueError, match="all-None Force"):
        _balance(
            (torch.ones(2, dtype=DTYPE),),
            force,
            (torch.tensor([0.0, 1.0], dtype=DTYPE),),
        )


def test_hessian_conflicting_energy_is_projected_and_norm_preserved() -> None:
    energy = (torch.tensor([1.0, -1.0], dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 1.0], dtype=DTYPE),)
    result = _balance(energy, (None,), hessian)

    expected_projected = torch.tensor([math.sqrt(2.0), 0.0], dtype=DTYPE)
    expected_hessian = torch.tensor([0.0, math.sqrt(2.0)], dtype=DTYPE)
    torch.testing.assert_close(
        result.projected_energy_gradients[0], expected_projected
    )
    torch.testing.assert_close(result.hessian_contributions[0], expected_hessian)
    torch.testing.assert_close(
        result.combined_gradients[0], expected_projected + expected_hessian
    )
    assert torch.linalg.vector_norm(result.projected_energy_gradients[0]).item() == (
        pytest.approx(torch.linalg.vector_norm(energy[0]).item())
    )
    block = result.audit["blocks"][0]
    assert block["projection"]["projection_applied"] is True
    assert block["hessian_nonconflict_passed"] is True
    assert block["assigned_hessian_dot"] >= 0.0


def test_weak_force_is_soft_and_does_not_project_energy() -> None:
    # Energy conflicts with Force but already agrees with Hessian.  A Force
    # hard guard would project it; the true weak-anchor rule must not.
    energy = (torch.tensor([-1.0, 1.0], dtype=DTYPE),)
    force = (torch.tensor([1.0, 0.0], dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 1.0], dtype=DTYPE),)
    result = _balance(energy, force, hessian, force_ratio=0.1)

    torch.testing.assert_close(result.projected_energy_gradients[0], energy[0])
    energy_norm = torch.linalg.vector_norm(energy[0]).item()
    assert torch.linalg.vector_norm(result.force_contributions[0]).item() == (
        pytest.approx(0.1 * energy_norm)
    )
    assert torch.linalg.vector_norm(result.hessian_contributions[0]).item() == (
        pytest.approx(energy_norm)
    )
    block = result.audit["blocks"][0]
    assert block["projection"]["projection_applied"] is False
    assert block["force_is_hard_gate"] is False
    assert block["assigned_force_dot"] < 0.0
    assert block["soft_force_compatible"] is False
    assert block["assigned_hessian_dot"] > 0.0
    assert result.audit["global_soft_force_compatible"] is False
    assert result.audit["global_hessian_nonconflict_passed"] is True


def test_positive_force_rescaling_does_not_cancel_explicit_anchor_ratio() -> None:
    energy = (torch.tensor([0.5, 1.0, -0.25], dtype=DTYPE),)
    force = (torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),)
    ordinary = _balance(energy, force, hessian, force_ratio=0.1)
    tiny = _balance(
        energy,
        (force[0] * 1.0e-200,),
        hessian,
        force_ratio=0.1,
    )

    torch.testing.assert_close(
        ordinary.combined_gradients[0], tiny.combined_gradients[0]
    )
    torch.testing.assert_close(
        ordinary.force_contributions[0], tiny.force_contributions[0]
    )
    assert torch.linalg.vector_norm(ordinary.force_contributions[0]).item() == (
        pytest.approx(
            0.1 * torch.linalg.vector_norm(energy[0]).item(), rel=2.0e-12
        )
    )


def test_force_ratio_changes_post_normalization_contribution() -> None:
    energy = (torch.tensor([1.0, 2.0], dtype=DTYPE),)
    force = (torch.tensor([1.0, 0.0], dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 1.0], dtype=DTYPE),)
    low = _balance(energy, force, hessian, force_ratio=0.1)
    high = _balance(energy, force, hessian, force_ratio=0.2)

    torch.testing.assert_close(
        high.force_contributions[0], 2.0 * low.force_contributions[0]
    )
    assert not torch.equal(high.combined_gradients[0], low.combined_gradients[0])


def test_weak_arm_allows_force_only_block_without_inventing_hessian() -> None:
    names = ("blocks.0.weight", "blocks.1.weight")
    groups = {names[0]: "blocks.0", names[1]: "blocks.1"}
    energy = (
        torch.tensor([1.0, 1.0], dtype=DTYPE),
        torch.tensor([2.0, 0.0], dtype=DTYPE),
    )
    force = (None, torch.tensor([1.0, 0.0], dtype=DTYPE))
    hessian = (torch.tensor([0.0, 1.0], dtype=DTYPE), None)
    result = _balance(
        energy,
        force,
        hessian,
        names=names,
        groups=groups,
        force_ratio=0.1,
    )

    assert result.hessian_contributions[1] is None
    assert result.force_contributions[0] is None
    assert result.force_contributions[1] is not None
    second = next(
        block for block in result.audit["blocks"] if block["group"] == "blocks.1"
    )
    assert second["projection"]["active_set"] == "none_no_hessian"
    assert second["hessian_nonconflict_passed"] is None
    assert result.audit["global_hessian_nonconflict_passed"] is True


def test_packing_splits_tensors_and_preserves_derivative_dead_energy() -> None:
    names = (
        "blocks.0.weight",
        "blocks.0.bias",
        "energy_head.network.4.bias",
    )
    groups = {
        names[0]: "blocks.0",
        names[1]: "blocks.0",
        names[2]: "energy_head",
    }
    energy = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE),
        torch.tensor([0.5, -0.5], dtype=DTYPE),
        torch.tensor([7.0], dtype=DTYPE),
    )
    hessian = (
        torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=DTYPE),
        None,
        None,
    )
    result = _balance(
        energy,
        (None, None, None),
        hessian,
        names=names,
        groups=groups,
    )

    assert result.combined_gradients[0].shape == energy[0].shape
    assert result.combined_gradients[1] is energy[1] or torch.equal(
        result.combined_gradients[1], energy[1]
    )
    assert torch.equal(result.combined_gradients[2], energy[2])
    assert result.audit["packed_parameter_tensor_count"] == 1
    assert result.audit["derivative_active_tensor_count"] == 1
    assert result.audit["derivative_dead_tensor_count"] == 2
    dead = next(
        block
        for block in result.audit["blocks"]
        if block["group"] == "energy_head"
    )
    assert dead["status"] == "derivative_dead"
    assert dead["derivative_dead_energy_preserved"] is True


def test_outputs_are_detached_from_input_graphs() -> None:
    source = torch.tensor([1.0, 2.0], dtype=DTYPE, requires_grad=True)
    energy = (source.square(),)
    hessian = (source * 3.0,)
    result = _balance(energy, (None,), hessian)

    for field in (
        result.combined_gradients,
        result.projected_energy_gradients,
        result.hessian_contributions,
        result.derivative_contributions,
    ):
        assert field[0] is not None
        assert field[0].grad_fn is None
        assert field[0].requires_grad is False


def test_production_float32_path_is_finite_and_hessian_safe() -> None:
    energy = (torch.tensor([-0.25, 0.5, 1.0], dtype=torch.float32),)
    force = (torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),)
    hessian = (torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),)
    result = _balance(energy, force, hessian, force_ratio=0.1)

    assert result.combined_gradients[0].dtype == torch.float32
    assert bool(torch.isfinite(result.combined_gradients[0]).all())
    assert result.audit["global_hessian_nonconflict_passed"] is True
    assert result.audit["numerical_tolerance"] == pytest.approx(
        256.0 * torch.finfo(torch.float32).eps
    )


def test_enabled_force_requires_a_reachable_gradient() -> None:
    with pytest.raises(RuntimeError, match="zero global Force"):
        _balance(
            (torch.ones(2, dtype=DTYPE),),
            (None,),
            (torch.tensor([0.0, 1.0], dtype=DTYPE),),
            force_ratio=0.1,
        )


def test_hessian_is_always_required_and_reachable() -> None:
    with pytest.raises(RuntimeError, match="zero global Hessian"):
        _balance(
            (torch.ones(2, dtype=DTYPE),),
            (None,),
            (None,),
        )


@pytest.mark.parametrize("ratio", [-1.0, float("nan"), float("inf")])
def test_invalid_force_ratio_fails_closed(ratio: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        _balance(
            (torch.ones(2, dtype=DTYPE),),
            (None,),
            (torch.tensor([0.0, 1.0], dtype=DTYPE),),
            force_ratio=ratio,
        )


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_hessian_ratio_fails_closed(ratio: float) -> None:
    match = "positive" if ratio == 0.0 else "finite and non-negative"
    with pytest.raises(ValueError, match=match):
        _balance(
            (torch.ones(2, dtype=DTYPE),),
            (None,),
            (torch.tensor([0.0, 1.0], dtype=DTYPE),),
            hessian_ratio=ratio,
        )


def test_large_soft_force_that_breaks_hessian_guard_fails_closed() -> None:
    energy = (torch.tensor([0.0, 1.0], dtype=DTYPE),)
    force = (torch.tensor([0.0, -1.0], dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 1.0], dtype=DTYPE),)
    with pytest.raises(FloatingPointError, match="conflicts with Hessian"):
        _balance(energy, force, hessian, force_ratio=3.0)


def test_boolean_ratios_are_rejected_as_non_numeric_controls() -> None:
    values = (torch.ones(2, dtype=DTYPE),)
    hessian = (torch.tensor([0.0, 1.0], dtype=DTYPE),)
    with pytest.raises(TypeError, match="real scalar"):
        _balance(values, (None,), hessian, force_ratio=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real scalar"):
        _balance(values, (None,), hessian, hessian_ratio=True)  # type: ignore[arg-type]
