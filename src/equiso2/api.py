"""Explicit paper architecture and small-system derivative evaluation."""
import torch
from .models.hess_so2_equiformer import HessSO2Config, HessSO2Equiformer
from .graph.topology import build_complete_topology


def paper_config() -> HessSO2Config:
    """508,640-parameter conservative configuration. No pretrained weights."""
    return HessSO2Config(channels=88, num_layers=3, num_rbf=12,
        radial_hidden=32, rank=2, num_foci=2, lmax=2, mmax=1,
        cutoff=5.0, skin=0.5, min_distance=1e-6, max_atomic_number=100,
        direct_force_head=False, invariant_scalar_feedback=True,
        invariant_scalar_feedback_epsilon=1e-6,
        invariant_scalar_feedback_mode="squared_norm",
        composition_energy_baseline="disabled")


def make_model(config=None, *, dtype=torch.float32, device="cpu"):
    """Construct a randomly initialized conservative model."""
    config = paper_config() if config is None else config
    if config.direct_force_head:
        raise ValueError("The public conservative factory requires direct_force_head=False")
    return HessSO2Equiformer(config).to(device=device, dtype=dtype)


def energy_force_hessian(model, atomic_numbers, positions, *, row_batch_size=1):
    """Evaluate E, F and raw Cartesian H for ONE isolated molecule.

    This reference implementation uses autograd and optionally batches Hessian
    rows. No mass weighting, projection, symmetrization or finite difference is
    applied. Outputs are detached. This is not a complete TS-search driver.
    """
    if isinstance(row_batch_size, bool) or not isinstance(row_batch_size, int) or row_batch_size < 1:
        raise ValueError("row_batch_size must be a positive integer")
    with torch.enable_grad():
        pos = positions.detach().clone().requires_grad_(True)
        batch = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
        topology = build_complete_topology(batch, model.config.cutoff, model.config.skin)
        out = model(atomic_numbers, pos, batch, topology=topology,
                    compute_forces=True, create_graph=True)
        gradient = -out.force_conservative.reshape(-1)
        rows = []
        for start in range(0, pos.numel(), row_batch_size):
            stop = min(start + row_batch_size, pos.numel())
            if row_batch_size == 1:
                row = torch.autograd.grad(gradient[start], pos, retain_graph=stop < pos.numel())[0]
                rows.append(row.reshape(1, -1))
            else:
                basis = pos.new_zeros((stop-start, pos.numel()))
                basis[torch.arange(stop-start, device=pos.device), torch.arange(start, stop, device=pos.device)] = 1
                row = torch.autograd.grad(gradient, pos, grad_outputs=basis,
                    is_grads_batched=True, retain_graph=stop < pos.numel())[0]
                rows.append(row.reshape(stop-start, -1))
        return out.energy.detach(), out.force_conservative.detach(), torch.cat(rows).detach()
