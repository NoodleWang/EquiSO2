from __future__ import annotations

import torch

from equiso2.curvature import (
    CurvatureResult,
    direct_force_vjp,
    exact_energy_hvp,
    finite_difference_force_jvp,
    finite_difference_hvp,
    minus_force_jacobian_vector,
    sample_coordinate_probes,
    sample_isotropic_probes,
)


DTYPE = torch.float64


def test_variable_batch_coordinate_and_isotropic_probes() -> None:
    ptr = torch.tensor([0, 1, 3])
    generator = torch.Generator().manual_seed(1234)
    coordinate = sample_coordinate_probes(
        ptr, [2, 1], generator=generator, dtype=DTYPE
    )

    assert coordinate.values.shape == (2, 9)
    assert torch.equal(
        coordinate.structure_mask,
        torch.tensor([[True, True], [True, False]]),
    )
    coordinate_ptr = 3 * ptr
    for q in range(2):
        for b in range(2):
            block = coordinate.values[q, coordinate_ptr[b] : coordinate_ptr[b + 1]]
            expected = 1.0 if coordinate.structure_mask[q, b] else 0.0
            assert torch.linalg.vector_norm(block).item() == expected

    isotropic = sample_isotropic_probes(
        ptr,
        [2, 1],
        generator=torch.Generator().manual_seed(5),
        dtype=DTYPE,
    )
    for q in range(2):
        for b in range(2):
            block = isotropic.values[q, coordinate_ptr[b] : coordinate_ptr[b + 1]]
            expected = torch.tensor(
                1.0 if isotropic.structure_mask[q, b] else 0.0, dtype=DTYPE
            )
            torch.testing.assert_close(torch.linalg.vector_norm(block), expected)


def test_direct_force_jvp_and_vjp_keep_orientation_explicit() -> None:
    matrix = torch.tensor(
        [[2.0, 1.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]],
        dtype=DTYPE,
    )
    positions = torch.tensor([[0.2, -0.3, 0.7]], dtype=DTYPE)
    probe = torch.tensor([1.0, -2.0, 0.5], dtype=DTYPE)

    def direct_force(x: torch.Tensor) -> torch.Tensor:
        return -(matrix @ x.reshape(-1)).reshape_as(x)

    jvp = minus_force_jacobian_vector(direct_force, positions, probe)
    vjp = direct_force_vjp(direct_force, positions, probe, create_graph=False)

    assert jvp.available_actions == ("minus_Jv",)
    assert vjp.available_actions == ("minus_JT_v",)
    torch.testing.assert_close(jvp.minus_Jv, matrix @ probe)
    torch.testing.assert_close(vjp.minus_JT_v, matrix.T @ probe)


def test_exact_energy_hvp_matches_quadratic_oracle_and_parameter_gradient() -> None:
    hessian = torch.tensor(
        [[3.0, 0.2, -0.1], [0.2, 2.0, 0.4], [-0.1, 0.4, 5.0]],
        dtype=DTYPE,
    )
    multiplier = torch.nn.Parameter(torch.tensor(1.7, dtype=DTYPE))
    positions = torch.tensor([[0.4, -0.2, 0.8]], dtype=DTYPE)
    probes = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.6, -0.8]], dtype=DTYPE
    )

    def energy(x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)
        return 0.5 * multiplier * flat @ hessian @ flat

    result = exact_energy_hvp(energy, positions, probes, create_graph=True)
    expected = multiplier.detach() * torch.einsum("ij,pj->pi", hessian, probes)
    torch.testing.assert_close(result.Hv.detach(), expected)
    result.Hv.sum().backward()
    expected_gradient = torch.einsum("ij,pj->", hessian, probes)
    torch.testing.assert_close(multiplier.grad, expected_gradient)


def test_centered_fd_distinguishes_generic_force_action_from_hvp() -> None:
    matrix = torch.tensor(
        [[2.0, 1.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]],
        dtype=DTYPE,
    )
    positions = torch.tensor([[0.2, -0.3, 0.7]], dtype=DTYPE)
    probe = torch.tensor([1.0, -2.0, 0.5], dtype=DTYPE)

    def force(x: torch.Tensor) -> torch.Tensor:
        return -(matrix @ x.reshape(-1)).reshape_as(x)

    generic = finite_difference_force_jvp(
        force, positions, probe, step_size=1e-4
    )
    conservative_claim = finite_difference_hvp(
        force, positions, probe, step_size=1e-4
    )

    torch.testing.assert_close(generic.minus_Jv, matrix @ probe)
    assert generic.Hv is None
    torch.testing.assert_close(conservative_claim.minus_Jv, matrix @ probe)
    torch.testing.assert_close(conservative_claim.Hv, matrix @ probe)


def test_fd_supports_per_structure_steps_and_parameter_backpropagation() -> None:
    ptr = torch.tensor([0, 1, 2])
    positions = torch.tensor(
        [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=DTYPE
    )
    probes = sample_isotropic_probes(
        ptr, 1, generator=torch.Generator().manual_seed(8), dtype=DTYPE
    )
    multiplier = torch.nn.Parameter(torch.tensor(2.5, dtype=DTYPE))

    def force(x: torch.Tensor) -> torch.Tensor:
        return -multiplier * x

    result = finite_difference_hvp(
        force,
        positions,
        probes,
        step_size=torch.tensor([[1e-4, 2e-4]], dtype=DTYPE),
    )
    torch.testing.assert_close(result.Hv, multiplier.detach() * probes.values)
    assert result.step_size.shape == (1, 2)
    result.Hv.sum().backward()
    torch.testing.assert_close(multiplier.grad, probes.values.sum())


def test_curvature_result_rejects_implicit_or_missing_orientation() -> None:
    probe = torch.ones(3)
    result = CurvatureResult(method="test", probe=probe, minus_Jv=probe)
    assert result.action("minus_Jv") is probe
    try:
        result.action("Hv")
    except ValueError as exc:
        assert "unavailable" in str(exc)
    else:  # pragma: no cover - explicit failure message is clearer than pytest.raises here
        raise AssertionError("Missing Hv should not be inferred from minus_Jv")
