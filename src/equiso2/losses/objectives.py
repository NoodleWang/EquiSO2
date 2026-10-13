"""Public per-structure force objective (not a pooled per-atom average)."""
import torch
from torch import Tensor
from equiso2.curvature.sampler import validate_ptr
from .energy_force import _scale_vector, _mask_vector


def force_loss(predicted: Tensor, target: Tensor, *, ptr: Tensor,
               scale: float | Tensor, valid_mask: Tensor | None = None) -> Tensor:
    """Mean_b ||F_b-F*_b||² / (3 N_b sigma_F²), over labelled structures.

    Inputs have shape [total_atoms, 3]. Missing structures are identified by
    valid_mask, with finite placeholder targets. All-missing batches return zero.
    """
    if predicted.ndim != 2 or predicted.shape[1] != 3 or target.shape != predicted.shape:
        raise ValueError("forces must have matching [total_atoms, 3] shapes")
    ptr = validate_ptr(ptr, n_atoms=predicted.shape[0])
    count = ptr.numel() - 1
    mask = _mask_vector(valid_mask, count, device=predicted.device, name="valid_mask")
    sigma = _scale_vector(scale, count, dtype=predicted.dtype, device=predicted.device, name="scale")
    difference = predicted - target.to(predicted)
    terms = [difference[int(ptr[b]):int(ptr[b+1])].square().mean() / sigma[b].square()
             for b in range(count) if bool(mask[b])]
    return torch.stack(terms).mean() if terms else predicted.sum() * 0.0
