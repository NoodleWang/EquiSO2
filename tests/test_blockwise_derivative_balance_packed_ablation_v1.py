from __future__ import annotations

import pytest
import torch

from equiso2.train.blockwise_derivative_balance_packed_ablation_v1 import (
    SCHEMA_VERSION,
    balance_blockwise_derivative_gradients_packed,
)
from equiso2.train.blockwise_derivative_balance_v1 import (
    balance_blockwise_derivative_gradients,
    coarse_parameter_groups,
)


DTYPE = torch.float32


def _assert_optional_gradients_close(actual, expected) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert (actual_value is None) == (expected_value is None)
        if actual_value is not None:
            assert expected_value is not None
            assert torch.allclose(
                actual_value, expected_value, rtol=2.0e-5, atol=2.0e-6
            )


def _paired_results():
    names = (
        "atom_embedding.weight",
        "blocks.0.first",
        "blocks.0.second",
        "blocks.1.weight",
        "energy_head.network.0.weight",
        "energy_head.network.4.bias",
    )
    groups = coarse_parameter_groups(names)
    energy = (
        torch.tensor([1.0, 2.0], dtype=DTYPE),
        torch.tensor([-0.3, 0.7, 0.2], dtype=DTYPE),
        torch.tensor([0.4], dtype=DTYPE),
        torch.tensor([0.8, -0.2], dtype=DTYPE),
        torch.tensor([0.5, 0.25], dtype=DTYPE),
        torch.tensor([11.0], dtype=DTYPE),
    )
    force = (
        torch.tensor([1.0, 0.0], dtype=DTYPE),
        torch.tensor([0.2, 0.1, -0.4], dtype=DTYPE),
        None,
        torch.tensor([0.3, 0.6], dtype=DTYPE),
        torch.tensor([0.0, 0.7], dtype=DTYPE),
        None,
    )
    hessian = (
        torch.tensor([0.0, 1.0], dtype=DTYPE),
        None,
        torch.tensor([0.9], dtype=DTYPE),
        torch.tensor([0.5, -0.1], dtype=DTYPE),
        torch.tensor([0.4, 0.0], dtype=DTYPE),
        torch.zeros(1, dtype=DTYPE),
    )
    ordinary = balance_blockwise_derivative_gradients(
        names,
        energy,
        force,
        hessian,
        parameter_groups=groups,
        derivative_target_ratio=0.75,
    )
    packed = balance_blockwise_derivative_gradients_packed(
        names,
        energy,
        force,
        hessian,
        parameter_groups=groups,
        derivative_target_ratio=0.75,
    )
    return names, energy, force, hessian, ordinary, packed


def test_packed_multiple_groups_match_original_controller() -> None:
    _, _, _, _, ordinary, packed = _paired_results()
    for actual, expected in (
        (packed.combined_gradients, ordinary.combined_gradients),
        (
            packed.projected_energy_gradients,
            ordinary.projected_energy_gradients,
        ),
        (packed.force_contributions, ordinary.force_contributions),
        (packed.hessian_contributions, ordinary.hessian_contributions),
        (
            packed.derivative_contributions,
            ordinary.derivative_contributions,
        ),
    ):
        _assert_optional_gradients_close(actual, expected)
    for metric in (
        "global_force_gradient_l2",
        "global_hessian_gradient_l2",
        "global_assigned_gradient_l2",
        "global_assigned_force_dot",
        "global_assigned_hessian_dot",
        "global_assigned_force_cosine",
        "global_assigned_hessian_cosine",
    ):
        assert packed.audit[metric] == pytest.approx(
            ordinary.audit[metric], rel=2.0e-6, abs=2.0e-7
        )


def test_none_and_derivative_dead_tensors_are_preserved_and_audited() -> None:
    names, energy, force, hessian, _, packed = _paired_results()
    bias_index = names.index("energy_head.network.4.bias")
    assert force[bias_index] is None
    assert hessian[bias_index] is not None
    assert not torch.any(hessian[bias_index] != 0.0)
    assert torch.equal(packed.combined_gradients[bias_index], energy[bias_index])
    assert torch.equal(
        packed.projected_energy_gradients[bias_index], energy[bias_index]
    )
    assert packed.force_contributions[bias_index] is None
    assert packed.hessian_contributions[bias_index] is None
    assert packed.derivative_contributions[bias_index] is None

    audit = packed.audit
    assert audit["schema_version"] == SCHEMA_VERSION
    assert audit["original_parameter_tensor_count"] == 6
    assert audit["packed_parameter_tensor_count"] == 4
    assert audit["derivative_active_tensor_count"] == 5
    assert audit["derivative_dead_tensor_count"] == 1
    assert audit["explicit_flag_host_sync_count"] == 1
    assert audit["explicit_metric_host_sync_count"] == 1
    assert audit["explicit_total_host_sync_count"] == 2
    energy_head = next(
        group for group in audit["groups"] if group["group"] == "energy_head"
    )
    assert energy_head["derivative_active_tensor_count"] == 1
    assert energy_head["derivative_dead_tensor_count"] == 1
    assert energy_head["derivative_dead_energy_preserved"] is True


def test_wholly_dead_group_is_preserved_without_a_synthetic_tensor() -> None:
    names = ("blocks.0.weight", "energy_head.network.4.bias")
    groups = {names[0]: "blocks.0", names[1]: "energy_head"}
    energy = (
        torch.tensor([1.0, 2.0], dtype=DTYPE),
        torch.tensor([9.0], dtype=DTYPE),
    )
    force = (torch.tensor([1.0, 0.0], dtype=DTYPE), None)
    hessian = (torch.tensor([0.0, 1.0], dtype=DTYPE), None)
    packed = balance_blockwise_derivative_gradients_packed(
        names,
        energy,
        force,
        hessian,
        parameter_groups=groups,
        derivative_target_ratio=1.0,
    )
    assert torch.equal(packed.combined_gradients[1], energy[1])
    assert packed.audit["packed_group_count"] == 1
    dead = next(
        group for group in packed.audit["groups"] if group["group"] == "energy_head"
    )
    assert dead["status"] == "derivative_dead"
    assert dead["base_controller_block"] is None


def test_returned_gradients_have_no_autograd_graph() -> None:
    names = ("blocks.0.first", "blocks.0.dead")
    groups = {name: "blocks.0" for name in names}
    energy = (
        torch.tensor([1.0, 2.0], dtype=DTYPE, requires_grad=True) * 2.0,
        torch.tensor([3.0], dtype=DTYPE, requires_grad=True) * 2.0,
    )
    force = (
        torch.tensor([1.0, 0.0], dtype=DTYPE, requires_grad=True) * 3.0,
        None,
    )
    hessian = (
        torch.tensor([0.0, 1.0], dtype=DTYPE, requires_grad=True) * 4.0,
        torch.zeros(1, dtype=DTYPE, requires_grad=True) * 4.0,
    )
    packed = balance_blockwise_derivative_gradients_packed(
        names,
        energy,
        force,
        hessian,
        parameter_groups=groups,
        derivative_target_ratio=1.0,
    )
    for field in (
        packed.combined_gradients,
        packed.projected_energy_gradients,
        packed.force_contributions,
        packed.hessian_contributions,
        packed.derivative_contributions,
    ):
        for gradient in field:
            if gradient is not None:
                assert gradient.requires_grad is False
                assert gradient.grad_fn is None


def test_batched_validation_rejects_nonfinite_and_misaligned_inputs() -> None:
    names = ("blocks.0.weight",)
    groups = {names[0]: "blocks.0"}
    finite = (torch.tensor([1.0, 2.0], dtype=DTYPE),)
    with pytest.raises(FloatingPointError, match="non-finite force"):
        balance_blockwise_derivative_gradients_packed(
            names,
            finite,
            (torch.tensor([float("nan"), 0.0], dtype=DTYPE),),
            (torch.tensor([0.0, 1.0], dtype=DTYPE),),
            parameter_groups=groups,
            derivative_target_ratio=1.0,
        )
    with pytest.raises(ValueError, match="shapes are not aligned"):
        balance_blockwise_derivative_gradients_packed(
            names,
            finite,
            (torch.ones(3, dtype=DTYPE),),
            finite,
            parameter_groups=groups,
            derivative_target_ratio=1.0,
        )
