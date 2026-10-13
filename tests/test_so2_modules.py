from __future__ import annotations

import torch
from e3nn import o3

from equiso2.irreps.layout import SO2EdgeTensor, SO2Layout
from equiso2.models.low_rank_so2 import LowRankSO2DegreeMixer
from equiso2.models.multi_focus import MultiFocusSO2Stack
from equiso2.models.smooth_attention import SmoothEnvelopeAttention


DTYPE = torch.float64


def _gauge_roll(values: torch.Tensor, layout: SO2Layout, angle: float) -> torch.Tensor:
    """Apply an e3nn local-y SO(2) roll to a compact edge tensor."""

    rotation = o3.matrix_y(torch.tensor(angle, dtype=values.dtype, device=values.device))
    pieces: list[torch.Tensor] = []
    offset = 0
    for degree in range(layout.lmax + 1):
        bound = min(degree, layout.mmax)
        retained = 2 * bound + 1
        full = o3.Irrep(degree, 1).D_from_matrix(rotation)
        keep = torch.arange(
            degree - bound,
            degree + bound + 1,
            dtype=torch.long,
            device=values.device,
        )
        compact_rotation = full.index_select(0, keep).index_select(1, keep)
        part = values[:, offset : offset + retained, :]
        pieces.append(torch.einsum("ij,ejc->eic", compact_rotation, part))
        offset += retained
    return torch.cat(pieces, dim=1)


def _edge_tensor(values: torch.Tensor, layout: SO2Layout) -> SO2EdgeTensor:
    return SO2EdgeTensor(values, layout, topology_id="topology", gauge_id="gauge")


def test_low_rank_mixer_shares_signed_m_kernel_and_commutes_with_gauge() -> None:
    torch.manual_seed(3)
    layout = SO2Layout(lmax=3, mmax=2)
    channels = 4
    mixer = LowRankSO2DegreeMixer(
        layout, channels=channels, radial_dim=5, rank=2, hidden_dim=7
    ).to(dtype=DTYPE)
    radial = torch.randn(3, 5, dtype=DTYPE)
    envelope = torch.tensor([0.2, 0.7, 1.0], dtype=DTYPE)
    values = torch.randn(3, layout.num_coefficients, channels, dtype=DTYPE)

    angle = 0.71
    rolled_input = _gauge_roll(values, layout, angle)
    mixed_then_rolled = _gauge_roll(
        mixer(_edge_tensor(values, layout), radial, envelope).values,
        layout,
        angle,
    )
    rolled_then_mixed = mixer(
        _edge_tensor(rolled_input, layout), radial, envelope
    ).values
    torch.testing.assert_close(
        rolled_then_mixed, mixed_then_rolled, rtol=1.0e-11, atol=1.0e-11
    )

    # Directly expose the shared-kernel contract: identical -m/+m degree lines
    # remain identical after the dynamic mixer.
    paired = torch.randn_like(values)
    for abs_order in range(1, layout.mmax + 1):
        negative = torch.tensor(layout.signed_order_indices(abs_order, -1))
        positive = torch.tensor(layout.signed_order_indices(abs_order, 1))
        paired[:, positive, :] = paired[:, negative, :]
    paired_output = mixer(_edge_tensor(paired, layout), radial, envelope).values
    for abs_order in range(1, layout.mmax + 1):
        negative = torch.tensor(layout.signed_order_indices(abs_order, -1))
        positive = torch.tensor(layout.signed_order_indices(abs_order, 1))
        torch.testing.assert_close(
            paired_output[:, negative, :],
            paired_output[:, positive, :],
            rtol=0.0,
            atol=0.0,
        )


def test_zero_envelope_disables_dynamic_edge_update() -> None:
    torch.manual_seed(9)
    layout = SO2Layout(lmax=2, mmax=1)
    values = torch.randn(2, layout.num_coefficients, 3, dtype=DTYPE)
    radial = torch.randn(2, 4, dtype=DTYPE)
    mixer = LowRankSO2DegreeMixer(
        layout, channels=3, radial_dim=4, rank=2, hidden_dim=6
    ).to(dtype=DTYPE)
    output = mixer(
        _edge_tensor(values, layout), radial, torch.zeros(2, dtype=DTYPE)
    )
    torch.testing.assert_close(output.values, values, rtol=0.0, atol=0.0)


def test_multi_focus_and_smooth_attention_commute_with_gauge_roll() -> None:
    torch.manual_seed(17)
    layout = SO2Layout(lmax=2, mmax=1)
    channels = 4
    values = torch.randn(4, layout.num_coefficients, channels, dtype=DTYPE)
    radial = torch.randn(4, 3, dtype=DTYPE)
    envelope = torch.tensor([0.0, 0.4, 0.8, 1.0], dtype=DTYPE)
    dst = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    angle = -0.43

    stack = MultiFocusSO2Stack(layout, channels=channels, num_foci=2).to(
        dtype=DTYPE
    )
    attention = SmoothEnvelopeAttention(
        layout, channels=channels, num_foci=2, radial_dim=3
    ).to(dtype=DTYPE)

    original_stacked = stack(_edge_tensor(values, layout))
    original_attended, original_weights = attention(
        original_stacked, radial, envelope, dst, num_nodes=2
    )
    rolled_values = _gauge_roll(values, layout, angle)
    rolled_stacked = stack(_edge_tensor(rolled_values, layout))
    rolled_attended, rolled_weights = attention(
        rolled_stacked, radial, envelope, dst, num_nodes=2
    )

    torch.testing.assert_close(
        rolled_stacked.values,
        _gauge_roll(original_stacked.values, layout, angle),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    torch.testing.assert_close(rolled_weights, original_weights, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(
        rolled_attended.values,
        _gauge_roll(original_attended.values, layout, angle),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    # A physically vanished edge is absent from both numerator and denominator.
    torch.testing.assert_close(
        original_weights[0], torch.zeros(2, dtype=DTYPE), rtol=0.0, atol=0.0
    )


def test_multi_focus_and_attention_accept_zero_edges() -> None:
    layout = SO2Layout(lmax=2, mmax=1)
    channels = 4
    values = torch.empty(
        (0, layout.num_coefficients, channels), dtype=DTYPE
    )
    edges = _edge_tensor(values, layout)
    stack = MultiFocusSO2Stack(layout, channels=channels, num_foci=2).to(
        dtype=DTYPE
    )
    attention = SmoothEnvelopeAttention(
        layout, channels=channels, num_foci=2, radial_dim=3
    ).to(dtype=DTYPE)

    stacked = stack(edges)
    attended, weights = attention(
        stacked,
        torch.empty((0, 3), dtype=DTYPE),
        torch.empty((0,), dtype=DTYPE),
        torch.empty((0,), dtype=torch.long),
        num_nodes=1,
    )

    assert stacked.values.shape == (0, layout.num_coefficients, channels)
    assert attended.values.shape == (0, layout.num_coefficients, channels)
    assert weights.shape == (0, 2)
