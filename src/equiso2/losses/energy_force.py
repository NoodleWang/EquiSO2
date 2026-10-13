"""Energy and force objectives with explicit physical-scale normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


PenaltyKind = Literal["mse", "mae", "huber"]


def robust_penalty(
    residual: Tensor,
    *,
    kind: PenaltyKind = "mse",
    huber_delta: float = 1.0,
) -> Tensor:
    """Apply a pointwise penalty without performing a reduction."""

    if kind == "mse":
        return residual.square()
    if kind == "mae":
        return residual.abs()
    if kind == "huber":
        if huber_delta <= 0:
            raise ValueError("`huber_delta` must be positive")
        absolute = residual.abs()
        delta = torch.as_tensor(huber_delta, dtype=residual.dtype, device=residual.device)
        return torch.where(
            absolute <= delta,
            0.5 * residual.square(),
            delta * (absolute - 0.5 * delta),
        )
    raise ValueError(f"Unknown penalty kind: {kind!r}")


def _scale_vector(
    scale: float | Tensor,
    length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> Tensor:
    value = torch.as_tensor(scale, dtype=dtype, device=device)
    if value.ndim == 0:
        value = value.expand(length)
    elif value.shape != (length,):
        raise ValueError(f"`{name}` must be scalar or have shape [{length}]")
    if not torch.all(torch.isfinite(value)) or torch.any(value <= 0):
        raise ValueError(f"`{name}` must contain finite positive values")
    return value


def _mask_vector(
    mask: Tensor | None,
    length: int,
    *,
    device: torch.device,
    name: str,
) -> Tensor:
    if mask is None:
        return torch.ones(length, dtype=torch.bool, device=device)
    value = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if value.shape != (length,):
        raise ValueError(f"`{name}` must have shape [{length}]")
    return value


def _masked_mean(values: Tensor, mask: Tensor, reference: Tensor) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError("Masked values and mask must have identical shapes")
    if torch.any(mask):
        return values[mask].mean()
    # Differentiable zero lets a trainer omit an unavailable label term without
    # inventing a target or breaking a shared backward call.
    return reference.sum() * 0.0


def energy_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    num_atoms: Tensor,
    scale: float | Tensor = 1.0,
    valid_mask: Tensor | None = None,
    penalty: PenaltyKind = "mse",
    huber_delta: float = 1.0,
) -> Tensor:
    r"""Dimensionless per-structure energy loss.

    Implements :math:`rho((\hat E-E)/(N\sigma_E))` from ``docs/whf.md``
    section 9.1. ``scale`` is fitted on training labels only and must be saved in
    the checkpoint by the caller.
    """

    if predicted.shape != target.shape:
        raise ValueError("Predicted and target energy tensors must have the same shape")
    pred = predicted.reshape(-1)
    tgt = target.to(device=pred.device, dtype=pred.dtype).reshape(-1)
    atoms = torch.as_tensor(num_atoms, dtype=pred.dtype, device=pred.device).reshape(-1)
    if atoms.shape != pred.shape or torch.any(atoms <= 0):
        raise ValueError("`num_atoms` must contain one positive value per structure")
    sigma = _scale_vector(
        scale,
        pred.numel(),
        dtype=pred.dtype,
        device=pred.device,
        name="scale",
    )
    mask = _mask_vector(
        valid_mask,
        pred.numel(),
        device=pred.device,
        name="valid_mask",
    )
    normalized = (pred - tgt) / (atoms * sigma)
    values = robust_penalty(normalized, kind=penalty, huber_delta=huber_delta)
    return _masked_mean(values, mask, pred)


def force_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    scale: float | Tensor = 1.0,
    atom_mask: Tensor | None = None,
    component_mask: Tensor | None = None,
    penalty: PenaltyKind = "mse",
    huber_delta: float = 1.0,
) -> Tensor:
    """Dimensionless force loss over valid atomic vectors or components.

    Without ``component_mask`` the residual is reduced to one L2 norm per atom,
    matching the vector form in ``docs/whf.md`` section 9.1. Partially labelled
    forces require ``component_mask`` and are reduced component-wise instead of
    pretending to be complete three-vectors.
    """

    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("Force tensors must have identical shape [n_atoms, 3]")
    tgt = target.to(device=predicted.device, dtype=predicted.dtype)
    n_atoms = predicted.shape[0]
    sigma = _scale_vector(
        scale,
        n_atoms,
        dtype=predicted.dtype,
        device=predicted.device,
        name="scale",
    )
    residual = predicted - tgt
    if component_mask is not None:
        if atom_mask is not None:
            raise ValueError("Use either atom_mask or component_mask, not both")
        mask = torch.as_tensor(
            component_mask, dtype=torch.bool, device=predicted.device
        )
        if mask.shape != predicted.shape:
            raise ValueError("`component_mask` must have shape [n_atoms,3]")
        normalized = residual / sigma[:, None]
        values = robust_penalty(normalized, kind=penalty, huber_delta=huber_delta)
        return _masked_mean(values, mask, predicted)

    mask = _mask_vector(
        atom_mask,
        n_atoms,
        device=predicted.device,
        name="atom_mask",
    )
    normalized_norm = torch.linalg.vector_norm(residual, dim=-1) / sigma
    values = robust_penalty(normalized_norm, kind=penalty, huber_delta=huber_delta)
    return _masked_mean(values, mask, predicted)


@dataclass(frozen=True)
class EnergyForceLossResult:
    total: Tensor
    energy: Tensor
    force: Tensor


def energy_force_loss(
    predicted_energy: Tensor,
    target_energy: Tensor,
    predicted_force: Tensor,
    target_force: Tensor,
    *,
    num_atoms: Tensor,
    energy_scale: float | Tensor = 1.0,
    force_scale: float | Tensor = 1.0,
    energy_valid_mask: Tensor | None = None,
    force_atom_mask: Tensor | None = None,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    energy_penalty: PenaltyKind = "mse",
    force_penalty: PenaltyKind = "mse",
    huber_delta: float = 1.0,
) -> EnergyForceLossResult:
    """Compute weighted E/F loss while retaining both diagnostic components."""

    if energy_weight < 0 or force_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    e_loss = energy_loss(
        predicted_energy,
        target_energy,
        num_atoms=num_atoms,
        scale=energy_scale,
        valid_mask=energy_valid_mask,
        penalty=energy_penalty,
        huber_delta=huber_delta,
    )
    f_loss = force_loss(
        predicted_force,
        target_force,
        scale=force_scale,
        atom_mask=force_atom_mask,
        penalty=force_penalty,
        huber_delta=huber_delta,
    )
    return EnergyForceLossResult(
        total=energy_weight * e_loss + force_weight * f_loss,
        energy=e_loss,
        force=f_loss,
    )


__all__ = [
    "EnergyForceLossResult",
    "PenaltyKind",
    "energy_force_loss",
    "energy_loss",
    "force_loss",
    "robust_penalty",
]
