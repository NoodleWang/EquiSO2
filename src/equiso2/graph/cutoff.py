"""Smooth radial functions used by every edge-dependent model path."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def c3_envelope(distance: Tensor, cutoff: float) -> Tensor:
    """Evaluate the C3 cutoff from the implementation assumption A4.

    This is the named implementation of the endpoint contract in
    ``whf.md``, Section 4.6.  For ``x = distance / cutoff`` it evaluates

    ``1 - 56*x**5 + 140*x**6 - 120*x**7 + 35*x**8`` for ``0 <= x < 1``
    and zero outside.  The value and first three one-sided derivatives vanish
    at the cutoff.

    Args:
        distance: Non-negative distances with arbitrary shape ``[...]``.
        cutoff: Positive physical cutoff in the same units.

    Returns:
        Envelope values with shape ``[...]`` and the input dtype/device.
    """

    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    x = distance / float(cutoff)
    x2 = x * x
    x4 = x2 * x2
    x5 = x4 * x
    polynomial = 1.0 + x5 * (-56.0 + x * (140.0 + x * (-120.0 + 35.0 * x)))
    inside = torch.logical_and(x >= 0.0, x < 1.0)
    return torch.where(inside, polynomial, torch.zeros_like(polynomial))


class GaussianRadialBasis(nn.Module):
    """Gaussian RBF expansion with the global C3 envelope contract.

    Input shape is ``[E]`` and output shape is ``[E, num_basis]``.  Multiplying
    the basis by the envelope ensures that radial conditioning itself has no
    un-gated cutoff bypass (``whf.md``, Section 4.6).
    """

    def __init__(self, num_basis: int, cutoff: float) -> None:
        super().__init__()
        if num_basis < 2:
            raise ValueError("num_basis must be at least 2")
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        self.num_basis = int(num_basis)
        self.cutoff = float(cutoff)
        centers = torch.linspace(0.0, self.cutoff, self.num_basis)
        spacing = self.cutoff / float(self.num_basis - 1)
        self.register_buffer("centers", centers, persistent=True)
        self.register_buffer(
            "inverse_width_squared",
            torch.tensor(1.0 / (spacing * spacing)),
            persistent=True,
        )

    def forward(self, distance: Tensor) -> Tensor:
        if distance.ndim != 1:
            raise ValueError(f"distance must have shape [E], got {distance.shape}")
        centers = self.centers.to(dtype=distance.dtype, device=distance.device)
        beta = self.inverse_width_squared.to(
            dtype=distance.dtype, device=distance.device
        )
        delta = distance.unsqueeze(1) - centers.unsqueeze(0)
        gaussian = torch.exp(-beta * delta.square())
        envelope = c3_envelope(distance, self.cutoff).unsqueeze(1)
        return gaussian * envelope

