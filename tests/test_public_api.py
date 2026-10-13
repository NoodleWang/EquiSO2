import runpy
from pathlib import Path
import pytest
import torch
from equiso2 import HessSO2Config, make_model, energy_force_hessian
from equiso2.losses import energy_loss, force_loss, hessian_vector_product_loss, full_hessian_loss


def test_paper_parameter_count_and_conservative_head():
    model = make_model()
    assert sum(p.numel() for p in model.parameters()) == 508640
    assert not model.config.direct_force_head


def test_variable_size_losses_weight_structures_not_atoms():
    ptr = torch.tensor([0, 1, 4])
    pred = torch.tensor([[1., 1., 1.], [2., 2., 2.], [2., 2., 2.], [2., 2., 2.]])
    assert force_loss(pred, torch.zeros_like(pred), ptr=ptr, scale=2.) == .625
    assert force_loss(pred, torch.zeros_like(pred), ptr=ptr, scale=2., valid_mask=torch.tensor([True, False])) == .25
    assert energy_loss(torch.tensor([1., 6.]), torch.zeros(2), num_atoms=ptr.diff(), scale=2.) == .625


def test_hvp_normalization_matches_full_matrix_in_expectation_exact_basis():
    # Average over coordinate unit probes is exactly isotropic.
    error = torch.diag(torch.arange(1., 7., dtype=torch.float64))
    ptr = torch.tensor([0, 2])
    lh = hessian_vector_product_loss(error.T, torch.zeros_like(error), ptr=ptr, scale=2.)
    full = full_hessian_loss([error], [torch.zeros_like(error)], ptr=ptr, scale=2.)
    torch.testing.assert_close(lh, full)


def test_hessian_row_batch_equivalence_and_force_finite_difference():
    torch.manual_seed(7)
    model = make_model(HessSO2Config(channels=8, num_layers=1, num_rbf=4,
        radial_hidden=8, direct_force_head=False), dtype=torch.float64).eval()
    z = torch.tensor([8, 1, 1])
    r = torch.tensor([[0., 0., 0.], [.96, .08, .03], [-.24, .93, -.06]], dtype=torch.float64)
    e, f, h = energy_force_hessian(model, z, r, row_batch_size=1)
    for size in [3, 63]:
        actual = energy_force_hessian(model, z, r, row_batch_size=size)
        for first, second in zip((e, f, h), actual):
            torch.testing.assert_close(first, second, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(h, h.T, rtol=1e-7, atol=1e-9)
    v = torch.randn_like(r)
    v = v / v.norm()
    batch = torch.zeros(3, dtype=torch.long)
    plus = model(z, r+1e-4*v, batch, create_graph=False).force_conservative
    minus = model(z, r-1e-4*v, batch, create_graph=False).force_conservative
    torch.testing.assert_close(h @ v.flatten(), -(plus-minus).flatten()/2e-4, rtol=1e-4, atol=1e-7)


def test_synthetic_training_step():
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/training_step.py"))
    losses, result = module["run"]()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert result.combined_gradients
