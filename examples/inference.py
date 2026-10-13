"""CPU-friendly E/F/H API demonstration. Outputs are NOT physical predictions."""
import torch
from equiso2 import make_model, energy_force_hessian

torch.manual_seed(7)
torch.set_num_threads(1)
model = make_model(dtype=torch.float64).eval()
z = torch.tensor([8, 1, 1])
r = torch.tensor([[0., 0., 0.], [0.96, 0., 0.], [-0.24, 0.93, 0.]], dtype=torch.float64)
energy, forces, hessian = energy_force_hessian(model, z, r, row_batch_size=3)
print("Random initialization, NOT a pretrained chemical potential")
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("E, F, H shapes:", energy.shape, forces.shape, hessian.shape)
print("Max Hessian asymmetry:", (hessian-hessian.T).abs().max().item())
