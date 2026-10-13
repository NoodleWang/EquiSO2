from __future__ import annotations

import torch
from e3nn import o3

from equiso2.graph.cutoff import c3_envelope
from equiso2.graph.geometry import (
    EdgeGeometry,
    deterministic_edge_frame,
    wigner_from_frame,
)
from equiso2.graph.topology import build_complete_topology
from equiso2.irreps.layout import SO2Layout, SO3Layout, SO3NodeTensor
from equiso2.irreps.rotation import lift_to_global, project_to_edge_gauge


DTYPE = torch.float64


def _derivatives_at(distance: float, cutoff: float) -> list[torch.Tensor]:
    coordinate = torch.tensor(distance, dtype=DTYPE, requires_grad=True)
    value = c3_envelope(coordinate, cutoff)
    derivatives = [value]
    for _ in range(3):
        value = torch.autograd.grad(value, coordinate, create_graph=True)[0]
        derivatives.append(value)
    return derivatives


def test_c3_envelope_value_and_first_three_derivatives_vanish_at_cutoff() -> None:
    cutoff = 5.0
    at_cutoff = _derivatives_at(cutoff, cutoff)
    outside = _derivatives_at(cutoff + 0.1, cutoff)
    for derivative in at_cutoff + outside:
        torch.testing.assert_close(
            derivative,
            torch.zeros_like(derivative),
            rtol=0.0,
            atol=2.0e-12,
        )

    # Verify the left branch rather than relying only on the torch.where value
    # at x=1. A fourfold endpoint zero gives delta**(4-k) for derivative k.
    small_delta = 0.01
    large_delta = 2.0 * small_delta
    small = _derivatives_at(cutoff - small_delta, cutoff)
    large = _derivatives_at(cutoff - large_delta, cutoff)
    expected_ratios = (16.0, 8.0, 4.0, 2.0)
    for derivative_small, derivative_large, expected in zip(
        small, large, expected_ratios, strict=True
    ):
        ratio = derivative_large.abs() / derivative_small.abs()
        torch.testing.assert_close(
            ratio,
            torch.tensor(expected, dtype=DTYPE),
            rtol=3.0e-2,
            atol=0.0,
        )


def test_deterministic_frame_is_right_handed_orthogonal_and_has_gradgrad() -> None:
    raw = torch.tensor(
        [[0.4, 0.2, 0.9], [-0.3, 0.95, 0.2], [0.7, -0.4, 0.5]],
        dtype=DTYPE,
        requires_grad=True,
    )
    unit = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    frame = deterministic_edge_frame(unit)
    repeated = deterministic_edge_frame(unit)

    identity = torch.eye(3, dtype=DTYPE).expand(raw.shape[0], 3, 3)
    gram = frame.transpose(-1, -2) @ frame
    torch.testing.assert_close(gram, identity, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(
        torch.linalg.det(frame),
        torch.ones(raw.shape[0], dtype=DTYPE),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(frame, repeated, rtol=0.0, atol=0.0)
    # The middle local axis is the edge direction by construction.
    torch.testing.assert_close(frame[:, :, 1], unit, rtol=0.0, atol=0.0)

    weights = torch.linspace(0.2, 1.7, frame.numel(), dtype=DTYPE).reshape_as(frame)
    first = torch.autograd.grad((frame * weights).sum(), raw, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), raw)[0]
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert first.abs().max() > 0
    assert second.abs().max() > 0


def test_edge_geometry_is_functional_and_full_project_lift_is_identity() -> None:
    torch.manual_seed(11)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.3, -0.2], [-0.4, 1.1, 0.7]],
        dtype=DTYPE,
        requires_grad=True,
    )
    batch = torch.zeros(positions.shape[0], dtype=torch.long)
    topology = build_complete_topology(batch, rcut=4.0, skin=0.5)
    node_layout = SO3Layout(lmax=2)
    edge_layout = SO2Layout(lmax=2, mmax=2)
    geometry_module = EdgeGeometry(node_layout, num_rbf=5, cutoff=4.0).to(
        dtype=DTYPE
    )
    geometry = geometry_module.materialize(positions, topology)
    geometry_again = geometry_module.materialize(positions, topology)

    assert geometry.requires_coordinate_grad
    torch.testing.assert_close(geometry.frame, geometry_again.frame, rtol=0.0, atol=0.0)
    for first, second in zip(geometry.wigner, geometry_again.wigner, strict=True):
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    values = torch.randn(
        positions.shape[0], node_layout.num_coefficients, 3, dtype=DTYPE
    )
    nodes = SO3NodeTensor(values, node_layout)
    local = project_to_edge_gauge(nodes, topology, geometry, edge_layout)
    lifted = lift_to_global(local, geometry)
    source = values.index_select(0, topology.src)
    torch.testing.assert_close(lifted, source, rtol=1.0e-10, atol=1.0e-11)

    # Exercise second derivatives through distances, frame, Wigner-D and RBF.
    scalar = geometry.radial.sum()
    for degree, wigner in enumerate(geometry.wigner[1:], start=1):
        coefficient = torch.linspace(
            0.1,
            1.0 + degree,
            wigner.numel(),
            dtype=DTYPE,
        ).reshape_as(wigner)
        scalar = scalar + (wigner * coefficient).sum()
    first = torch.autograd.grad(scalar, positions, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), positions)[0]
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_axial_edges_have_finite_wigner_first_and_second_derivatives() -> None:
    """Exact y-axis edges must not hit an Euler-angle coordinate singularity."""

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.2, 0.0], [0.8, -0.4, 0.3]],
        dtype=DTYPE,
        requires_grad=True,
    )
    batch = torch.zeros(positions.shape[0], dtype=torch.long)
    topology = build_complete_topology(batch, rcut=4.0, skin=0.5)
    geometry = EdgeGeometry(SO3Layout(lmax=2), num_rbf=5, cutoff=4.0).to(
        dtype=DTYPE
    ).materialize(positions, topology)
    coefficients = [
        torch.linspace(0.1, 1.0, matrix.numel(), dtype=DTYPE).reshape_as(matrix)
        for matrix in geometry.wigner
    ]
    scalar = sum(
        (matrix * coefficient).sum()
        for matrix, coefficient in zip(
            geometry.wigner, coefficients, strict=True
        )
    )
    first = torch.autograd.grad(scalar, positions, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), positions)[0]

    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_direct_wigner_matches_e3nn_representation_away_from_singularities() -> None:
    axes = torch.tensor(
        [[0.3, 0.4, 0.5], [-0.2, 0.7, 0.4]], dtype=DTYPE
    )
    axes = axes / torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
    frames = o3.axis_angle_to_matrix(
        axes, torch.tensor([0.7, 1.2], dtype=DTYPE)
    )
    actual = wigner_from_frame(frames, lmax=2)

    for degree in (1, 2):
        expected = o3.Irrep(degree, 1).D_from_matrix(frames)
        torch.testing.assert_close(
            actual[degree], expected, rtol=1.0e-5, atol=1.0e-6
        )
