"""Per-structure-normalized Hessian action and full-matrix losses."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from equiso2.curvature import ActionName, CurvatureResult
from equiso2.curvature.sampler import ProbeBatch, validate_ptr

from .energy_force import _scale_vector


def _action_matrix(value: Tensor, *, name: str) -> tuple[Tensor, bool]:
    if value.ndim == 1:
        return value.unsqueeze(0), True
    if value.ndim == 2:
        return value, False
    raise ValueError(f"`{name}` must have shape [D] or [P,D]")


def _probe_structure_mask(
    mask: Tensor | None,
    *,
    n_probe: int,
    n_structure: int,
    device: torch.device,
) -> Tensor:
    if mask is None:
        return torch.ones((n_probe, n_structure), dtype=torch.bool, device=device)
    value = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if value.shape == (n_structure,):
        value = value[None, :].expand(n_probe, n_structure)
    if value.shape != (n_probe, n_structure):
        raise ValueError("Probe mask must have shape [B] or [P,B]")
    return value


def hessian_action_loss(
    predicted_action: Tensor,
    target_action: Tensor,
    *,
    ptr: Tensor,
    scale: float | Tensor = 1.0,
    probe_mask: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> Tensor:
    r"""Squared random-curvature loss normalized within each structure.

    For structure ``b`` this computes

    .. math::
       (s_b d_b \sigma_{H,b}^2)^{-1}
       \sum_q \|\widehat H_bv_{bq}-H_bv_{bq}\|_2^2,

    then averages over structures that have at least one valid probe. This is the
    size-fair objective in ``docs/whf.md`` section 5.1.
    """

    predicted, _ = _action_matrix(predicted_action, name="predicted_action")
    target, _ = _action_matrix(target_action, name="target_action")
    if predicted.shape != target.shape:
        raise ValueError("Predicted and target actions must have identical shapes")
    ptr_t = validate_ptr(ptr).to(predicted.device)
    total_dof = 3 * int(ptr_t[-1].item())
    if predicted.shape[1] != total_dof:
        raise ValueError("Action width does not match 3*ptr[-1]")
    n_probe, n_structure = predicted.shape[0], ptr_t.numel() - 1
    mask = _probe_structure_mask(
        probe_mask,
        n_probe=n_probe,
        n_structure=n_structure,
        device=predicted.device,
    )
    if valid_mask is not None:
        mask = mask & _probe_structure_mask(
            valid_mask,
            n_probe=n_probe,
            n_structure=n_structure,
            device=predicted.device,
        )
    sigma = _scale_vector(
        scale,
        n_structure,
        dtype=predicted.dtype,
        device=predicted.device,
        name="scale",
    )
    difference = predicted - target.to(device=predicted.device, dtype=predicted.dtype)
    coordinate_ptr = 3 * ptr_t
    structure_losses: list[Tensor] = []
    for b in range(n_structure):
        active = mask[:, b]
        count = int(active.sum().item())
        if count == 0:
            continue
        start = int(coordinate_ptr[b].item())
        stop = int(coordinate_ptr[b + 1].item())
        d_b = stop - start
        squared = difference[active, start:stop].square().sum()
        structure_losses.append(squared / (count * d_b * sigma[b].square()))
    if not structure_losses:
        return predicted.sum() * 0.0
    return torch.stack(structure_losses).mean()


def curvature_result_loss(
    result: CurvatureResult,
    target_action: Tensor,
    *,
    ptr: Tensor | None = None,
    orientation: ActionName = "Hv",
    scale: float | Tensor = 1.0,
    probe_mask: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Loss for one explicitly selected :class:`CurvatureResult` orientation."""

    ptr_value = ptr if ptr is not None else result.ptr
    if ptr_value is None:
        raise ValueError("An atom ptr is required for per-structure normalization")
    result_mask = result.valid_mask
    if probe_mask is not None and result_mask is not None:
        action_device = result.action(orientation).device
        result_mask = torch.as_tensor(
            probe_mask, dtype=torch.bool, device=action_device
        ) & torch.as_tensor(result_mask, dtype=torch.bool, device=action_device)
    elif probe_mask is not None:
        result_mask = probe_mask
    return hessian_action_loss(
        result.action(orientation),
        target_action,
        ptr=ptr_value,
        scale=scale,
        probe_mask=result_mask,
        valid_mask=valid_mask,
    )


def _hessian_sequence(
    hessians: Tensor | Sequence[Tensor],
    ptr: Tensor,
    *,
    name: str,
) -> list[Tensor]:
    n_structure = ptr.numel() - 1
    coordinate_ptr = 3 * ptr
    if isinstance(hessians, Tensor):
        total_dof = int(coordinate_ptr[-1].item())
        if hessians.ndim == 2:
            if n_structure == 1 and hessians.shape == (total_dof, total_dof):
                return [hessians]
            if hessians.shape == (total_dof, total_dof):
                return [
                    hessians[
                        int(coordinate_ptr[b]) : int(coordinate_ptr[b + 1]),
                        int(coordinate_ptr[b]) : int(coordinate_ptr[b + 1]),
                    ]
                    for b in range(n_structure)
                ]
        if hessians.ndim == 3 and hessians.shape[0] == n_structure:
            result = []
            for b in range(n_structure):
                d_b = int((coordinate_ptr[b + 1] - coordinate_ptr[b]).item())
                if hessians.shape[1] < d_b or hessians.shape[2] < d_b:
                    raise ValueError(f"Padded `{name}` is too small for structure {b}")
                result.append(hessians[b, :d_b, :d_b])
            return result
        raise ValueError(
            f"`{name}` tensor must be block [D,D], single [d,d], or padded [B,d,d]"
        )
    result = list(hessians)
    if len(result) != n_structure:
        raise ValueError(f"`{name}` must contain one Hessian per structure")
    for b, value in enumerate(result):
        d_b = int((coordinate_ptr[b + 1] - coordinate_ptr[b]).item())
        if value.shape != (d_b, d_b):
            raise ValueError(f"`{name}[{b}]` must have shape [{d_b},{d_b}]")
    return result


def reference_hessian_actions(
    hessians: Tensor | Sequence[Tensor],
    probes: Tensor | ProbeBatch,
    *,
    ptr: Tensor | None = None,
    transpose: bool = False,
) -> Tensor:
    """Apply reference Hessian blocks as ``Hv`` or ``H.T @ v``.

    ``transpose=True`` is the explicit target orientation for a direct-force VJP,
    even when cleaned reference labels are expected to be symmetric.
    """

    if isinstance(probes, ProbeBatch):
        probe_values = probes.values
        ptr_value = probes.ptr if ptr is None else ptr
        squeeze = False
    else:
        probe_values, squeeze = _action_matrix(probes, name="probes")
        ptr_value = ptr
    if ptr_value is None:
        raise ValueError("An atom ptr is required")
    ptr_t = validate_ptr(ptr_value).to(probe_values.device)
    if probe_values.shape[1] != 3 * int(ptr_t[-1].item()):
        raise ValueError("Probe width does not match 3*ptr[-1]")
    blocks = _hessian_sequence(hessians, ptr_t, name="hessians")
    result = torch.zeros_like(probe_values)
    coordinate_ptr = 3 * ptr_t
    for b, block in enumerate(blocks):
        start = int(coordinate_ptr[b].item())
        stop = int(coordinate_ptr[b + 1].item())
        matrix = block.to(device=result.device, dtype=result.dtype)
        if transpose:
            matrix = matrix.transpose(-1, -2)
        result[:, start:stop] = torch.einsum(
            "ij,pj->pi", matrix, probe_values[:, start:stop]
        )
    return result.squeeze(0) if squeeze else result


def full_hessian_loss(
    predicted: Tensor | Sequence[Tensor],
    target: Tensor | Sequence[Tensor],
    *,
    ptr: Tensor,
    scale: float | Tensor = 1.0,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Normalized full-Hessian Frobenius MSE, averaged by structure."""

    ptr_t = validate_ptr(ptr)
    pred_blocks = _hessian_sequence(predicted, ptr_t, name="predicted")
    target_blocks = _hessian_sequence(target, ptr_t, name="target")
    n_structure = ptr_t.numel() - 1
    device = pred_blocks[0].device
    dtype = pred_blocks[0].dtype
    sigma = _scale_vector(
        scale, n_structure, dtype=dtype, device=device, name="scale"
    )
    if valid_mask is None:
        mask = torch.ones(n_structure, dtype=torch.bool, device=device)
    else:
        mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
        if mask.shape != (n_structure,):
            raise ValueError("valid_mask must have shape [B]")
    losses: list[Tensor] = []
    for b, (pred, tgt) in enumerate(zip(pred_blocks, target_blocks, strict=True)):
        if not bool(mask[b]):
            continue
        difference = pred - tgt.to(device=pred.device, dtype=pred.dtype)
        d_b = pred.shape[0]
        losses.append(difference.square().sum() / (d_b * d_b * sigma[b].square()))
    if not losses:
        return sum(block.sum() for block in pred_blocks) * 0.0
    return torch.stack(losses).mean()


__all__ = [
    "curvature_result_loss",
    "full_hessian_loss",
    "hessian_action_loss",
    "reference_hessian_actions",
]
