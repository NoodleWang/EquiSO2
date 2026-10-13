"""Public interface to the research implementation of gradient coordination."""
import torch
from .train.blockwise_derivative_balance_v1 import (
    assign_parameter_gradients, coarse_parameter_groups,
)
from .train.blockwise_derivative_balance_packed_ablation_v1 import balance_blockwise_derivative_gradients_packed
from .train.blockwise_sobolev_balance_packed_v1 import balance_blockwise_sobolev_gradients_packed


def coordinate_gradients(model, energy_loss, force_loss, hessian_loss, *,
                         mode="EH-soft-F", force_ratio=0.1, hessian_ratio=1.0):
    """Assign detached coordinated .grad values, but do NOT step the optimizer.

    EH: Hessian protected, force omitted. EH-soft-F: Hessian protected, force
    soft. EFH: force and Hessian protected, with both ratios fixed at one.
    Non-conflict is checked before any clipping or adaptive optimizer update.
    """
    if mode not in ("EH", "EH-soft-F", "EFH"):
        raise ValueError("mode must be EH, EH-soft-F or EFH")
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names, parameters = zip(*named)
    def gradient(loss):
        if loss is None:
            return (None,) * len(parameters)
        return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    if energy_loss is None or hessian_loss is None or (mode != "EH" and force_loss is None):
        raise ValueError("Selected mode requires energy, Hessian, and (unless EH) force losses")
    if mode == "EFH" and (force_ratio != 1.0 or hessian_ratio != 1.0):
        raise ValueError("EFH requires explicitly setting force_ratio=hessian_ratio=1")
    args = (names, gradient(energy_loss), gradient(None if mode == "EH" else force_loss), gradient(hessian_loss))
    options = dict(parameter_groups=coarse_parameter_groups(names),
                   force_target_ratio=0.0 if mode == "EH" else force_ratio,
                   hessian_target_ratio=hessian_ratio)
    if mode == "EFH":
        result = balance_blockwise_derivative_gradients_packed(*args, derivative_target_ratio=1.0, **options)
    else:
        result = balance_blockwise_sobolev_gradients_packed(*args, **options)
    assign_parameter_gradients(parameters, result.combined_gradients)
    return result
