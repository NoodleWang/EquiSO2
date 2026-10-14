# EquiSO2

A lightweight implementation of a conservative equivariant interatomic
potential and curvature-prioritized learning. The default configuration contains
**508,640 trainable parameters**. Energies, forces and Hessians derive from a
single scalar energy function.

## Scope

Included:

- Edge-local, low-rank SO(2) interactions, multi-focus mixing and invariant readout.
- Structure-balanced normalized energy, force and Hessian fitting losses.
- Unit random probes and centered force-difference Hessian–vector products.
- Blockwise gradient coordination for EH, EH-soft-F and EFH.
- Energy/force/full-Hessian inference and one-step synthetic training examples.
- CPU tests for symmetry, derivatives, normalization and gradient protection.

Not included: pretrained weights, datasets, full training campaigns, cluster
submission scripts, manuscript files, Gaussian/ORCA interfaces or TS-search
infrastructure. The examples use **random weights and synthetic inputs** and
must not be interpreted as chemically meaningful predictions or reproduction
of the reported reaction-search success rates.

## Install and run

Use Python 3.10 or newer in an isolated environment. Install the PyTorch build
appropriate for your CPU or CUDA environment, then run from this repository:

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 python examples/inference.py
OMP_NUM_THREADS=1 python examples/training_step.py
OMP_NUM_THREADS=1 python -m pytest -q
```

The initial compatibility target is PyTorch 2.8 and e3nn 0.6. NumPy is used
for topology metadata. No dataset download or GPU is required for the tests.

## Model API

```python
import torch
from equiso2 import make_model, energy_force_hessian

model = make_model(dtype=torch.float64).eval()  # random initialization
z = torch.tensor([8, 1, 1])
r = torch.tensor([[0., 0., 0.], [.96, 0., 0.], [-.24, .93, 0.]], dtype=torch.float64)
energy, force, hessian = energy_force_hessian(model, z, r, row_batch_size=3)
```

This helper evaluates one isolated molecule and returns energy `[1]`, force
`[N,3]` and an unprojected Cartesian Hessian `[3N,3N]`. Atom-major ordering is
used. Coordinates, model parameters and atomic numbers must share a device,
and floating-point inputs must match the model dtype. Do not use inference
mode for derivative evaluation.

For molecular batches, call the model directly with concatenated atom arrays
and contiguous zero-based `batch` indices. `ptr` contains cumulative **atom**
counts, not Cartesian component counts. The loss functions support unequal
molecule sizes.

With the training conventions used in the study, coordinates are in angstrom,
energy is atomization energy in eV, force is in eV/angstrom and Hessian entries
are in eV/angstrom². Units are not converted automatically. Energy reference
conventions, normalization scales and architecture settings must accompany any
future checkpoint release. An embedding table supporting atomic numbers up to
100 does not imply training coverage or accuracy for all elements.

The public factory selects the conservative paper configuration. Legacy
`HessSO2Config` and `HessSO2Equiformer` names remain internally to preserve
parameter naming. Their historical constructor defaults differ from the paper
configuration. Use `make_model()` or `paper_config()` explicitly.

## Losses and gradient coordination

For a labelled structure with N atoms and d = 3N Cartesian components:

| Objective | Per-structure squared error |
| --- | --- |
| Energy | `(E - E_ref)^2 / (N^2 sigma_E^2)` |
| Force | `sum((F - F_ref)^2) / (d sigma_F^2)` |
| Full Hessian | `sum((H - H_ref)^2) / (d^2 sigma_H^2)` |
| Stochastic Hessian | `mean_q sum((H_pred v_q - H_ref v_q)^2) / (d sigma_H^2)` |

Each loss averages structures equally, over structures with its own labels.
The energy scale is the training-set sample standard deviation of energy per
atom. Force and Hessian scales are pooled component root-mean-square
magnitudes, without mean subtraction. Supply positive scales computed from
training labels, preserve them in checkpoints and reuse them for validation
and resumed training. Missing labels use explicit masks and finite placeholder
targets. The public `equiso2.losses.force_loss` is the structure-balanced
objective. The retained legacy function in `losses.energy_force` has additional
reduction conventions and is not the public force-loss entry point.

For unit isotropic probes, averaging exact Hessian–vector products reproduces
the normalized full-Hessian objective in expectation. The training example
approximates predicted products by `-[F(R+h*v)-F(R-h*v)]/(2*h)`, with `h=1e-3`
angstrom. Both displaced evaluations share discrete topology but recompute
continuous geometry. Finite probe counts introduce sampling variance, and
finite differences introduce discretization error. Full-Hessian inference uses
automatic differentiation, not this finite-difference estimator.

| Mode | Force ratio | Hessian ratio | Protected tasks |
| --- | --- | --- | --- |
| EH | 0 | 1 | Hessian |
| EH-soft-F | 0.1 | 1 | Hessian |
| EFH | 1 | 1 | Force and Hessian |

`coordinate_gradients` combines parameter gradients, rather than optimizing a
fixed weighted sum of these losses. Within the embedding, each interaction
block and the energy head, it projects the derivative-active energy gradient
into protected-task non-conflict half-spaces, restores its norm, and adds
norm-scaled derivative directions. Other parameter tensors retain their energy
gradients. The combined direction is checked before assignment. Degenerate or
infeasible cases raise exceptions rather than silently ignoring protection.

Ratios specify derivative contributions relative to the active energy-gradient
norm, not scalar loss weights. EH-soft-F does not impose a force-specific
non-conflict constraint. AdamW momentum, adaptive scaling and finite steps mean
these pre-optimizer checks do not guarantee monotonically decreasing losses.

```python
from equiso2.controller import coordinate_gradients

optimizer.zero_grad(set_to_none=True)
result = coordinate_gradients(model, loss_E, loss_F, loss_H, mode="EH-soft-F")
torch.nn.utils.clip_grad_norm_(model.parameters(), 20., error_if_nonfinite=True)
optimizer.step()
```

For EH, supply `None` for force loss. For EFH, also specify `force_ratio=1.0`.
Use the low-level controllers for precomputed or streamed parameter gradients.
The convenience interface retains graphs for a small single-process example.
It is not the distributed or memory-optimized campaign trainer.

## Repository layout

```text
src/equiso2/models, graph, irreps   model and differentiable geometry
src/equiso2/curvature             probe sampling and derivative operators
src/equiso2/losses                fitting objectives
src/equiso2/train                 original standalone controller implementations
src/equiso2/api.py                explicit model factory and derivative helper
src/equiso2/controller.py         convenience gradient-coordination interface
examples/                        executable synthetic examples
tests/                           self-contained tests
```

References to internal design notes in legacy docstrings are historical
annotations, not runtime dependencies. This README specifies the public
interface. This source release does not reproduce the optimized model
service or establish the manuscript's hardware timing results.

## Authors

Hengfang Wang and Jin-Cheng Liu

Corresponding author: Jin-Cheng Liu (liujincheng@nankai.edu.cn)


## License

Copyright (c) 2026 (Hengfang Wang, Jin-Cheng Liu).

Unless otherwise noted, the original EquiSO2 source code, examples, tests
and documentation in this repository are licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License
(CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

See [LICENSE](LICENSE) for the full license terms.

Third-party materials and dependencies remain subject to their respective
licenses. Dependencies are installed separately, not vendored. This license
notice does not replace any third-party license terms.

For licensing inquiries, please contact liujincheng@nankai.edu.cn.

## Acknowledgements

The implementation builds on equivariant tensor representations,
edge-aligned SO(2) processing, and the DPA4 low-rank and multi-focus design
ideas discussed in the accompanying manuscript.
