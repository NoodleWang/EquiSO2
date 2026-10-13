from __future__ import annotations

import math

import pytest
import torch

from equiso2.graph.topology import build_complete_topology
from equiso2.irreps.layout import SO3Layout, SO3NodeTensor
from equiso2.models import HessSO2Config, HessSO2Equiformer
from equiso2.models.block import EquivariantNodeGate


DTYPE = torch.float64


def _rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis.unbind()
    zero = x * 0.0
    cross = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return identity + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _small_model(
    *,
    direct_force_head: bool,
    invariant_scalar_feedback: bool = False,
    invariant_scalar_feedback_epsilon: float = 1.0e-8,
    invariant_scalar_feedback_mode: str = "sqrt_norm",
    num_layers: int = 1,
) -> HessSO2Equiformer:
    torch.manual_seed(4)
    config = HessSO2Config(
        lmax=2,
        mmax=1,
        channels=8,
        num_layers=num_layers,
        num_rbf=5,
        radial_hidden=12,
        rank=2,
        num_foci=2,
        cutoff=4.0,
        skin=0.5,
        direct_force_head=direct_force_head,
        invariant_scalar_feedback=invariant_scalar_feedback,
        invariant_scalar_feedback_epsilon=invariant_scalar_feedback_epsilon,
        invariant_scalar_feedback_mode=invariant_scalar_feedback_mode,
    )
    return HessSO2Equiformer(config).to(dtype=DTYPE)


def _molecule() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    atomic_numbers = torch.tensor([1, 6, 8], dtype=torch.long)
    positions = torch.tensor(
        [[-0.7, 0.1, 0.2], [0.9, -0.3, 0.4], [0.2, 1.1, -0.5]],
        dtype=DTYPE,
    )
    batch = torch.zeros(3, dtype=torch.long)
    return atomic_numbers, positions, batch


@pytest.mark.parametrize("invariant_scalar_feedback", [False, True])
def test_model_energy_is_rotation_translation_invariant_and_forces_covary(
    invariant_scalar_feedback: bool,
) -> None:
    model = _small_model(
        direct_force_head=True,
        invariant_scalar_feedback=invariant_scalar_feedback,
    ).eval()
    atomic_numbers, positions, batch = _molecule()
    rotation = _rotation(torch.tensor([0.3, -0.5, 0.8], dtype=DTYPE), 0.73)
    translation = torch.tensor([1.2, -0.4, 0.7], dtype=DTYPE)
    transformed_positions = positions @ rotation.T + translation

    original = model(
        atomic_numbers,
        positions,
        batch,
        compute_forces=True,
        create_graph=True,
    )
    transformed = model(
        atomic_numbers,
        transformed_positions,
        batch,
        compute_forces=True,
        create_graph=True,
    )

    torch.testing.assert_close(
        transformed.energy, original.energy, rtol=1.0e-10, atol=1.0e-11
    )
    torch.testing.assert_close(
        transformed.force_conservative,
        original.force_conservative @ rotation.T,
        rtol=1.0e-9,
        atol=1.0e-10,
    )
    # Route-D is auxiliary but must remain an equivariant vector head.
    torch.testing.assert_close(
        transformed.force_direct,
        original.force_direct @ rotation.T,
        rtol=1.0e-7,
        atol=2.0e-9,
    )


@pytest.mark.parametrize("invariant_scalar_feedback", [False, True])
def test_model_exact_second_derivative_is_finite_symmetric_and_translation_null(
    invariant_scalar_feedback: bool,
) -> None:
    model = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=invariant_scalar_feedback,
    ).eval()
    atomic_numbers, positions, batch = _molecule()
    topology = build_complete_topology(
        batch, model.config.cutoff, model.config.skin
    )

    def scalar_energy(coordinate: torch.Tensor) -> torch.Tensor:
        energy, _, _, _ = model.energy_features(
            atomic_numbers, coordinate, batch, topology
        )
        return energy.sum()

    hessian = torch.autograd.functional.hessian(scalar_energy, positions).reshape(9, 9)
    assert torch.isfinite(hessian).all()
    torch.testing.assert_close(
        hessian, hessian.T, rtol=1.0e-9, atol=1.0e-10
    )
    translation_modes = torch.eye(3, dtype=DTYPE).repeat(3, 1)
    torch.testing.assert_close(
        hessian @ translation_modes,
        torch.zeros((9, 3), dtype=DTYPE),
        rtol=0.0,
        atol=1.0e-10,
    )


def test_force_objective_backpropagates_mixed_second_derivatives_to_parameters() -> None:
    model = _small_model(direct_force_head=False).train()
    atomic_numbers, positions, batch = _molecule()
    topology = build_complete_topology(
        batch, model.config.cutoff, model.config.skin
    )
    coordinate = positions.clone().requires_grad_(True)
    output = model(
        atomic_numbers,
        coordinate,
        batch,
        topology,
        compute_forces=True,
        create_graph=True,
    )
    objective = output.energy.square().sum() + output.force_conservative.square().sum()
    objective.backward()

    gradients = {
        name: parameter.grad for name, parameter in model.named_parameters()
    }
    assert all(gradient is not None for gradient in gradients.values())
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())
    dynamic_gradient = gradients["blocks.0.dynamic.kernel_heads.0.0.weight"]
    assert dynamic_gradient.abs().max() > 0


@pytest.mark.parametrize("invariant_scalar_feedback", [False, True])
def test_single_atom_zero_edge_model_has_finite_energy_and_zero_derivatives(
    invariant_scalar_feedback: bool,
) -> None:
    model = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=invariant_scalar_feedback,
    ).eval()
    atomic_numbers = torch.tensor([1], dtype=torch.long)
    positions = torch.tensor([[0.2, -0.4, 0.7]], dtype=DTYPE)
    batch = torch.zeros(1, dtype=torch.long)

    output = model(
        atomic_numbers,
        positions,
        batch,
        compute_forces=True,
        create_graph=True,
    )

    assert output.energy.shape == (1,)
    assert torch.isfinite(output.energy).all()
    assert output.topology.num_edges == 0
    torch.testing.assert_close(
        output.force_conservative,
        torch.zeros((1, 3), dtype=DTYPE),
        rtol=0.0,
        atol=0.0,
    )

    def scalar_energy(coordinate: torch.Tensor) -> torch.Tensor:
        energy, _, _, _ = model.energy_features(
            atomic_numbers, coordinate, batch, output.topology
        )
        return energy.sum()

    hessian = torch.autograd.functional.hessian(scalar_energy, positions)
    torch.testing.assert_close(
        hessian,
        torch.zeros((1, 3, 1, 3), dtype=DTYPE),
        rtol=0.0,
        atol=0.0,
    )


def test_disabled_scalar_feedback_preserves_old_state_dict_and_behavior() -> None:
    torch.manual_seed(91)
    implicit = _small_model(direct_force_head=False).eval()
    torch.manual_seed(91)
    explicit = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=False,
    ).eval()

    implicit_state = implicit.state_dict()
    explicit_state = explicit.state_dict()
    assert implicit_state.keys() == explicit_state.keys()
    for name in implicit_state:
        assert implicit_state[name].shape == explicit_state[name].shape
        torch.testing.assert_close(
            implicit_state[name], explicit_state[name], rtol=0.0, atol=0.0
        )
    explicit.load_state_dict(implicit_state, strict=True)
    assert implicit.blocks[0].gate.scalar_map.weight.shape == (8, 8)

    atomic_numbers, positions, batch = _molecule()
    implicit_output = implicit(
        atomic_numbers, positions, batch, compute_forces=True, create_graph=False
    )
    explicit_output = explicit(
        atomic_numbers, positions, batch, compute_forces=True, create_graph=False
    )
    torch.testing.assert_close(
        implicit_output.energy, explicit_output.energy, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        implicit_output.force_conservative,
        explicit_output.force_conservative,
        rtol=0.0,
        atol=0.0,
    )
    assert implicit.provenance["invariant_scalar_feedback"] is False
    assert (
        implicit.provenance["scalar_feedback_initialization"]
        == "legacy_linear_v1"
    )
    assert implicit.provenance["invariant_scalar_feedback_epsilon"] == 1.0e-8
    assert implicit.provenance["invariant_scalar_feedback_mode"] == "sqrt_norm"


@pytest.mark.parametrize(
    "epsilon",
    [0.0, -1.0e-8, float("nan"), float("inf"), -float("inf")],
)
def test_scalar_feedback_epsilon_must_be_finite_and_positive(epsilon: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        HessSO2Config(invariant_scalar_feedback_epsilon=epsilon)


@pytest.mark.parametrize("epsilon", [True, "1e-6", None])
def test_scalar_feedback_epsilon_must_be_numeric(epsilon: object) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        HessSO2Config(invariant_scalar_feedback_epsilon=epsilon)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["norm", "square", "", "SQRT_NORM"])
def test_scalar_feedback_mode_must_be_supported(mode: str) -> None:
    with pytest.raises(ValueError, match="sqrt_norm.*squared_norm"):
        HessSO2Config(invariant_scalar_feedback_mode=mode)


@pytest.mark.parametrize("mode", [True, None, 1])
def test_scalar_feedback_mode_must_be_string(mode: object) -> None:
    with pytest.raises(TypeError, match="must be str"):
        HessSO2Config(invariant_scalar_feedback_mode=mode)  # type: ignore[arg-type]


def test_legacy_checkpoint_without_feedback_epsilon_loads_strictly() -> None:
    legacy = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=False,
    ).eval()
    legacy_config = dict(legacy.config.__dict__)
    legacy_config.pop("invariant_scalar_feedback_epsilon")
    legacy_config.pop("invariant_scalar_feedback_mode")

    restored = HessSO2Equiformer(HessSO2Config(**legacy_config)).to(dtype=DTYPE)
    restored.load_state_dict(legacy.state_dict(), strict=True)
    assert restored.config.invariant_scalar_feedback_epsilon == 1.0e-8
    assert restored.config.invariant_scalar_feedback_mode == "sqrt_norm"

    atomic_numbers, positions, batch = _molecule()
    expected = legacy(
        atomic_numbers, positions, batch, compute_forces=True, create_graph=False
    )
    actual = restored(
        atomic_numbers, positions, batch, compute_forces=True, create_graph=False
    )
    torch.testing.assert_close(actual.energy, expected.energy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual.force_conservative,
        expected.force_conservative,
        rtol=0.0,
        atol=0.0,
    )


def _last_block_m1_gradient_norms(
    model: HessSO2Equiformer,
) -> dict[tuple[str, str], float]:
    atomic_numbers = torch.tensor([1, 6, 8, 7], dtype=torch.long)
    positions = torch.tensor(
        [
            [-0.9, 0.2, 0.3],
            [0.8, -0.4, 0.5],
            [0.1, 1.2, -0.6],
            [0.5, 0.3, 1.1],
        ],
        dtype=DTYPE,
    )
    batch = torch.zeros(4, dtype=torch.long)
    output = model(
        atomic_numbers,
        positions,
        batch,
        compute_forces=True,
        create_graph=True,
    )
    assert output.force_conservative is not None
    named_parameters = list(model.named_parameters())
    parameters = [parameter for _, parameter in named_parameters]
    energy_gradients = torch.autograd.grad(
        output.energy.sum(), parameters, retain_graph=True, allow_unused=True
    )
    force_target = torch.linspace(
        -0.2,
        0.3,
        output.force_conservative.numel(),
        dtype=DTYPE,
    ).reshape_as(output.force_conservative)
    force_objective = (output.force_conservative - force_target).square().sum()
    probe = torch.linspace(
        -0.3,
        0.4,
        output.force_conservative.numel(),
        dtype=DTYPE,
    ).reshape_as(output.force_conservative)
    force_jacobian_action = torch.autograd.grad(
        (output.force_conservative * probe).sum(),
        output.positions,
        create_graph=True,
        retain_graph=True,
    )[0]
    hvp_target = torch.linspace(
        -0.1,
        0.2,
        force_jacobian_action.numel(),
        dtype=DTYPE,
    ).reshape_as(force_jacobian_action)
    hvp_objective = (force_jacobian_action - hvp_target).square().sum()
    force_gradients = torch.autograd.grad(
        force_objective, parameters, retain_graph=True, allow_unused=True
    )
    hvp_gradients = torch.autograd.grad(
        hvp_objective, parameters, allow_unused=True
    )
    result: dict[tuple[str, str], float] = {}
    for module in ("dynamic", "static"):
        if module == "dynamic":
            selected = lambda name: (
                name.startswith("blocks.1.dynamic.")
                and (
                    ".kernel_heads.1." in name
                    or ".channel_factors.1" in name
                    or ".residual_scales.1" in name
                )
            )
        else:
            selected = lambda name: (
                name.startswith("blocks.1.static.") and "_m1" in name
            )
        indices = [
            index
            for index, (name, _) in enumerate(named_parameters)
            if selected(name)
        ]
        assert indices, f"no final-block m=1 {module} parameters found"
        for objective, gradients in (
            ("energy", energy_gradients),
            ("force", force_gradients),
            ("hvp", hvp_gradients),
        ):
            squared = 0.0
            for index in indices:
                gradient = gradients[index]
                if gradient is not None:
                    assert torch.isfinite(gradient).all()
                    squared += float(gradient.detach().square().sum())
            result[(module, objective)] = math.sqrt(squared)
    return result


def test_scalar_feedback_initialization_is_function_preserving_and_common_rng() -> None:
    disabled = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=False,
        num_layers=2,
    ).train()
    enabled = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=True,
        invariant_scalar_feedback_epsilon=1.0e-6,
        num_layers=2,
    ).train()

    disabled_state = disabled.state_dict()
    enabled_state = enabled.state_dict()
    assert disabled_state.keys() == enabled_state.keys()
    for name, disabled_value in disabled_state.items():
        enabled_value = enabled_state[name]
        if name.endswith("gate.scalar_map.weight"):
            assert disabled_value.shape == (8, 8)
            assert enabled_value.shape == (8, 24)
            torch.testing.assert_close(
                enabled_value[:, :8], disabled_value, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                enabled_value[:, 8:],
                torch.zeros_like(enabled_value[:, 8:]),
                rtol=0.0,
                atol=0.0,
            )
        else:
            assert enabled_value.shape == disabled_value.shape, name
            torch.testing.assert_close(
                enabled_value, disabled_value, rtol=0.0, atol=0.0
            )

    atomic_numbers, positions, batch = _molecule()
    disabled_output = disabled(
        atomic_numbers, positions, batch, compute_forces=True, create_graph=True
    )
    enabled_output = enabled(
        atomic_numbers, positions, batch, compute_forces=True, create_graph=True
    )
    torch.testing.assert_close(
        enabled_output.energy, disabled_output.energy, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        enabled_output.force_conservative,
        disabled_output.force_conservative,
        rtol=0.0,
        atol=0.0,
    )
    assert enabled.provenance["invariant_scalar_feedback"] is True
    assert (
        enabled.provenance["scalar_feedback_initialization"]
        == "function_preserving_common_rng_zero_columns_v1"
    )
    assert enabled.provenance["invariant_scalar_feedback_epsilon"] == 1.0e-6
    assert enabled.provenance["invariant_scalar_feedback_mode"] == "sqrt_norm"


def test_squared_feedback_initialization_matches_disabled_and_sqrt_modes() -> None:
    disabled = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=False,
        num_layers=2,
    ).train()
    sqrt_norm = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=True,
        invariant_scalar_feedback_mode="sqrt_norm",
        num_layers=2,
    ).train()
    squared_norm = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=True,
        invariant_scalar_feedback_mode="squared_norm",
        num_layers=2,
    ).train()

    disabled_state = disabled.state_dict()
    sqrt_state = sqrt_norm.state_dict()
    squared_state = squared_norm.state_dict()
    assert sqrt_state.keys() == squared_state.keys() == disabled_state.keys()
    for name, disabled_value in disabled_state.items():
        sqrt_value = sqrt_state[name]
        squared_value = squared_state[name]
        torch.testing.assert_close(squared_value, sqrt_value, rtol=0.0, atol=0.0)
        if name.endswith("gate.scalar_map.weight"):
            torch.testing.assert_close(
                squared_value[:, :8], disabled_value, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                squared_value[:, 8:],
                torch.zeros_like(squared_value[:, 8:]),
                rtol=0.0,
                atol=0.0,
            )
        else:
            torch.testing.assert_close(
                squared_value, disabled_value, rtol=0.0, atol=0.0
            )

    atomic_numbers, positions, batch = _molecule()
    outputs = [
        model(
            atomic_numbers,
            positions,
            batch,
            compute_forces=True,
            create_graph=True,
        )
        for model in (disabled, sqrt_norm, squared_norm)
    ]
    for output in outputs[1:]:
        torch.testing.assert_close(
            output.energy, outputs[0].energy, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            output.force_conservative,
            outputs[0].force_conservative,
            rtol=0.0,
            atol=0.0,
        )
    assert squared_norm.provenance["invariant_scalar_feedback_mode"] == "squared_norm"


@pytest.mark.parametrize("feedback_epsilon", [1.0e-8, 1.0e-6])
def test_scalar_feedback_epsilon_is_equivariant_and_has_finite_second_derivatives(
    feedback_epsilon: float,
) -> None:
    model = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=True,
        invariant_scalar_feedback_epsilon=feedback_epsilon,
    ).eval()
    # Activate the feedback path; at its function-preserving initialization the
    # configured epsilon would otherwise be multiplied by exactly zero.
    with torch.no_grad():
        for block in model.blocks:
            block.gate.scalar_map.weight[:, 8:].fill_(2.0e-2)

    atomic_numbers, positions, batch = _molecule()
    rotation = _rotation(torch.tensor([0.3, -0.5, 0.8], dtype=DTYPE), 0.73)
    translation = torch.tensor([1.2, -0.4, 0.7], dtype=DTYPE)
    transformed_positions = positions @ rotation.T + translation
    original_energy = model(
        atomic_numbers, positions, batch, compute_forces=False
    ).energy
    transformed_energy = model(
        atomic_numbers, transformed_positions, batch, compute_forces=False
    ).energy
    torch.testing.assert_close(
        transformed_energy, original_energy, rtol=1.0e-10, atol=1.0e-11
    )

    topology = build_complete_topology(
        batch, model.config.cutoff, model.config.skin
    )

    def scalar_energy(coordinate: torch.Tensor) -> torch.Tensor:
        energy, _, _, _ = model.energy_features(
            atomic_numbers, coordinate, batch, topology
        )
        return energy.sum()

    hessian = torch.autograd.functional.hessian(scalar_energy, positions).reshape(9, 9)
    assert torch.isfinite(hessian).all()
    torch.testing.assert_close(hessian, hessian.T, rtol=1.0e-9, atol=1.0e-10)


def test_squared_feedback_is_invariant_and_has_finite_symmetric_hessian() -> None:
    model = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=True,
        invariant_scalar_feedback_mode="squared_norm",
    ).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.gate.scalar_map.weight[:, 8:].fill_(2.0e-2)

    atomic_numbers, positions, batch = _molecule()
    rotation = _rotation(torch.tensor([0.3, -0.5, 0.8], dtype=DTYPE), 0.73)
    translation = torch.tensor([1.2, -0.4, 0.7], dtype=DTYPE)
    transformed_positions = positions @ rotation.T + translation
    original = model(
        atomic_numbers, positions, batch, compute_forces=True, create_graph=True
    )
    transformed = model(
        atomic_numbers,
        transformed_positions,
        batch,
        compute_forces=True,
        create_graph=True,
    )
    torch.testing.assert_close(
        transformed.energy, original.energy, rtol=1.0e-10, atol=1.0e-11
    )
    torch.testing.assert_close(
        transformed.force_conservative,
        original.force_conservative @ rotation.T,
        rtol=1.0e-9,
        atol=1.0e-10,
    )

    topology = build_complete_topology(
        batch, model.config.cutoff, model.config.skin
    )

    def scalar_energy(coordinate: torch.Tensor) -> torch.Tensor:
        energy, _, _, _ = model.energy_features(
            atomic_numbers, coordinate, batch, topology
        )
        return energy.sum()

    hessian = torch.autograd.functional.hessian(scalar_energy, positions).reshape(9, 9)
    assert torch.isfinite(hessian).all()
    torch.testing.assert_close(hessian, hessian.T, rtol=1.0e-9, atol=1.0e-10)


def _zero_feedback_second_derivative(feedback_epsilon: float) -> torch.Tensor:
    layout = SO3Layout(lmax=1)
    gate = EquivariantNodeGate(
        layout,
        channels=1,
        invariant_scalar_feedback=True,
        invariant_scalar_feedback_epsilon=feedback_epsilon,
    ).to(dtype=DTYPE)
    with torch.no_grad():
        gate.scalar_map.weight.zero_()
        gate.scalar_map.weight[0, 1] = 1.0
        gate.scalar_map.bias.zero_()

    def scalar_output(coefficient: torch.Tensor) -> torch.Tensor:
        zero = coefficient * 0.0
        values = torch.stack((zero, coefficient, zero, zero)).reshape(1, 4, 1)
        nodes = SO3NodeTensor(values, layout)
        return gate(nodes).values[0, 0, 0]

    origin = torch.zeros((), dtype=DTYPE)
    return torch.autograd.functional.hessian(scalar_output, origin)


def test_squared_feedback_has_finite_exact_zero_point_curvature() -> None:
    layout = SO3Layout(lmax=1)
    gate = EquivariantNodeGate(
        layout,
        channels=1,
        invariant_scalar_feedback=True,
        invariant_scalar_feedback_mode="squared_norm",
    ).to(dtype=DTYPE)
    with torch.no_grad():
        gate.scalar_map.weight.zero_()
        gate.scalar_map.weight[0, 1] = 1.0
        gate.scalar_map.bias.zero_()

    def scalar_output(coefficient: torch.Tensor) -> torch.Tensor:
        zero = coefficient * 0.0
        values = torch.stack((zero, coefficient, zero, zero)).reshape(1, 4, 1)
        return gate(SO3NodeTensor(values, layout)).values[0, 0, 0]

    origin = torch.zeros((), dtype=DTYPE)
    curvature = torch.autograd.functional.hessian(scalar_output, origin)
    assert torch.isfinite(curvature)
    # silu'(0)=1/2 and d2(x^2)/dx2=2, hence the exact composition is 1.
    torch.testing.assert_close(
        curvature, torch.ones((), dtype=DTYPE), rtol=0.0, atol=0.0
    )


def test_squared_feedback_does_not_depend_on_feedback_epsilon() -> None:
    layout = SO3Layout(lmax=2)
    gates = [
        EquivariantNodeGate(
            layout,
            channels=2,
            invariant_scalar_feedback=True,
            invariant_scalar_feedback_epsilon=epsilon,
            invariant_scalar_feedback_mode="squared_norm",
        ).to(dtype=DTYPE)
        for epsilon in (1.0e-12, 1.0e2)
    ]
    gates[1].load_state_dict(gates[0].state_dict(), strict=True)
    with torch.no_grad():
        for gate in gates:
            gate.scalar_map.weight[:, 2:].fill_(3.0e-2)
    values = torch.linspace(
        -0.7,
        0.9,
        2 * layout.num_coefficients * 2,
        dtype=DTYPE,
    ).reshape(2, layout.num_coefficients, 2)
    nodes = SO3NodeTensor(values, layout)
    outputs = [gate(nodes).values for gate in gates]
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0.0, atol=0.0)


def test_larger_feedback_epsilon_reduces_zero_point_high_order_scale() -> None:
    scale_1e8 = _zero_feedback_second_derivative(1.0e-8).abs()
    scale_1e6 = _zero_feedback_second_derivative(1.0e-6).abs()
    assert torch.isfinite(scale_1e8)
    assert torch.isfinite(scale_1e6)
    assert scale_1e8 > 0.0
    assert scale_1e6 < scale_1e8
    # sqrt(x^2 + eps) has zero-point curvature proportional to eps^-1/2;
    # the smooth SiLU composition perturbs but does not erase the ~10x ratio.
    assert scale_1e6 < 0.2 * scale_1e8


@pytest.mark.parametrize("feedback_mode", ["sqrt_norm", "squared_norm"])
def test_scalar_feedback_update_opens_final_m1_energy_and_force_paths(
    feedback_mode: str,
) -> None:
    disabled = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=False,
        num_layers=2,
    ).train()
    enabled = _small_model(
        direct_force_head=False,
        invariant_scalar_feedback=True,
        invariant_scalar_feedback_mode=feedback_mode,
        num_layers=2,
    ).train()

    # A zero feedback map preserves the old function, so upstream final-block
    # m=1 paths remain closed before its first update.
    disabled_norms = _last_block_m1_gradient_norms(disabled)
    enabled_norms = _last_block_m1_gradient_norms(enabled)
    for module in ("dynamic", "static"):
        for objective in ("energy", "force", "hvp"):
            assert disabled_norms[(module, objective)] == 0.0
            assert enabled_norms[(module, objective)] == 0.0

    atomic_numbers, positions, batch = _molecule()
    output = enabled(
        atomic_numbers, positions, batch, compute_forces=True, create_graph=True
    )
    assert output.force_conservative is not None
    target = torch.linspace(
        -0.2,
        0.3,
        output.force_conservative.numel(),
        dtype=DTYPE,
    ).reshape_as(output.force_conservative)
    objectives = {
        "energy": output.energy.square().sum(),
        "force": (output.force_conservative - target).square().sum(),
    }
    probe = torch.linspace(
        -0.3,
        0.4,
        output.force_conservative.numel(),
        dtype=DTYPE,
    ).reshape_as(output.force_conservative)
    force_jacobian_action = torch.autograd.grad(
        (output.force_conservative * probe).sum(),
        output.positions,
        create_graph=True,
        retain_graph=True,
    )[0]
    objectives["hvp"] = force_jacobian_action.square().sum()
    feedback_weight = enabled.blocks[-1].gate.scalar_map.weight
    feedback_gradients = []
    for objective in objectives.values():
        feedback_gradient = torch.autograd.grad(
            objective, feedback_weight, retain_graph=True
        )[0][:, 8:]
        assert torch.isfinite(feedback_gradient).all()
        assert feedback_gradient.abs().max() > 0.0
        feedback_gradients.append(feedback_gradient)

    # One explicit SGD update activates only the new columns.  Common weights
    # stay paired while the intended non-scalar -> scalar route becomes live.
    with torch.no_grad():
        feedback_weight[:, 8:].add_(
            torch.stack(feedback_gradients).sum(dim=0), alpha=-1.0e-3
        )
    activated_norms = _last_block_m1_gradient_norms(enabled)
    for module in ("dynamic", "static"):
        for objective_name in ("energy", "force", "hvp"):
            assert activated_norms[(module, objective_name)] > 0.0
