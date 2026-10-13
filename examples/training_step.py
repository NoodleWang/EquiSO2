"""One synthetic EH-soft-F step. This is a wiring test, not scientific training."""
import torch
from equiso2 import HessSO2Config, make_model
from equiso2.controller import coordinate_gradients
from equiso2.curvature.sampler import sample_isotropic_probes
from equiso2.curvature.finite_difference import finite_difference_hvp
from equiso2.graph.topology import build_complete_topology
from equiso2.losses import energy_loss, force_loss, hessian_vector_product_loss, reference_hessian_vector_products


def run():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    # Small configuration keeps the demonstration quick. make_model() without
    # config constructs the paper architecture instead.
    config = HessSO2Config(channels=8, num_layers=1, num_rbf=4, radial_hidden=8,
        direct_force_head=False, invariant_scalar_feedback=True,
        invariant_scalar_feedback_mode="squared_norm")
    model = make_model(config, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    z = torch.tensor([8, 1, 1])
    r = torch.tensor([[0., 0., 0.], [.96, .08, .03], [-.24, .93, -.06]], dtype=torch.float64)
    batch = torch.zeros(3, dtype=torch.long)
    ptr = torch.tensor([0, 3])
    topology = build_complete_topology(batch, config.cutoff, config.skin)
    # Synthetic quadratic labels, not electronic-structure reference data.
    target_e = r.square().sum().reshape(1) * 0.5
    target_f = -r
    target_h = [torch.eye(9, dtype=r.dtype)]
    out = model(z, r, batch, topology=topology, create_graph=True)
    probes = sample_isotropic_probes(ptr, num_probes=1, dtype=r.dtype)
    def forces(displaced):
        return model(z, displaced, batch, topology=topology, create_graph=True).force_conservative
    predicted = finite_difference_hvp(forces, r, probes, step_size=1e-3).action("Hv")
    reference = reference_hessian_vector_products(target_h, probes)
    # Unit scales are placeholders ONLY for this synthetic test. Real training
    # must supply scales computed on training labels and save them with weights.
    le = energy_loss(out.energy, target_e, num_atoms=ptr.diff(), scale=1.)
    lf = force_loss(out.force_conservative, target_f, ptr=ptr, scale=1.)
    lh = hessian_vector_product_loss(predicted, reference, ptr=ptr,
        scale=1., probe_mask=probes.structure_mask)
    optimizer.zero_grad(set_to_none=True)
    result = coordinate_gradients(model, le, lf, lh, mode="EH-soft-F")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 20., error_if_nonfinite=True)
    optimizer.step()
    return {"energy": float(le.detach()), "force": float(lf.detach()), "hessian": float(lh.detach())}, result


if __name__ == "__main__":
    losses, _ = run()
    print("One synthetic EH-soft-F update completed:", losses)
