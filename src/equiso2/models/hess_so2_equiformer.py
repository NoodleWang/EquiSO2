"""Standalone conservative HessSO2-Equiformer C1 prototype."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from equiso2.graph.geometry import EdgeGeometry
from equiso2.graph.topology import EdgeTopology, build_complete_topology
from equiso2.irreps.layout import SO2Layout, SO3Layout, SO3NodeTensor

from .block import HessSO2Block
from .direct_force_head import DirectForceHead
from .energy_head import AtomicEnergyHead
from .output import PotentialOutput


@dataclass(frozen=True)
class HessSO2Config:
    """Executable defaults for choices not fixed by ``whf.md`` (ambiguity A6)."""

    lmax: int = 2
    mmax: int = 1
    channels: int = 32
    num_layers: int = 2
    num_rbf: int = 16
    radial_hidden: int = 64
    rank: int = 2
    num_foci: int = 2
    cutoff: float = 5.0
    skin: float = 0.5
    min_distance: float = 1.0e-6
    max_atomic_number: int = 100
    direct_force_head: bool = True
    invariant_scalar_feedback: bool = False
    invariant_scalar_feedback_epsilon: float = 1.0e-8
    invariant_scalar_feedback_mode: str = "sqrt_norm"
    composition_energy_baseline: str = "disabled"

    def __post_init__(self) -> None:
        if self.lmax < 1:
            raise ValueError("lmax must be at least 1")
        if self.mmax < 0 or self.mmax > self.lmax:
            raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
        if self.channels <= 0 or self.channels % self.num_foci != 0:
            raise ValueError("channels must be positive and divisible by num_foci")
        if self.num_layers <= 0 or self.num_rbf < 2 or self.rank <= 0:
            raise ValueError("num_layers/rank must be positive and num_rbf >= 2")
        if self.cutoff <= 0.0 or self.skin < 0.0:
            raise ValueError("cutoff must be positive and skin non-negative")
        if not isinstance(self.invariant_scalar_feedback, bool):
            raise TypeError("invariant_scalar_feedback must be bool")
        if not isinstance(self.invariant_scalar_feedback_mode, str):
            raise TypeError("invariant_scalar_feedback_mode must be str")
        if self.invariant_scalar_feedback_mode not in {"sqrt_norm", "squared_norm"}:
            raise ValueError(
                "invariant_scalar_feedback_mode must be 'sqrt_norm' or "
                "'squared_norm'"
            )
        if self.composition_energy_baseline not in {
            "disabled",
            "fixed_train_residual_v1",
        }:
            raise ValueError(
                "composition_energy_baseline must be disabled or "
                "fixed_train_residual_v1"
            )
        if isinstance(self.invariant_scalar_feedback_epsilon, bool) or not isinstance(
            self.invariant_scalar_feedback_epsilon, (int, float)
        ):
            raise TypeError("invariant_scalar_feedback_epsilon must be a real number")
        if (
            not math.isfinite(float(self.invariant_scalar_feedback_epsilon))
            or self.invariant_scalar_feedback_epsilon <= 0.0
        ):
            raise ValueError(
                "invariant_scalar_feedback_epsilon must be finite and positive"
            )


class HessSO2Equiformer(nn.Module):
    """Global SO(3) nodes with edge-local low-rank SO(2) computation.

    The conservative path implements ``whf.md``, Sections 4.1--4.8:
    atom scalars -> differentiable edge gauge -> A1 degree mixing ->
    multi-focus static SO(2) -> C3 attention -> scalar atomic energies.  Forces
    are obtained only as ``-grad_R E``.  The optional direct head is returned
    separately for route-D comparisons.
    """

    def __init__(self, config: HessSO2Config | None = None) -> None:
        super().__init__()
        self.config = config or HessSO2Config()
        self.node_layout = SO3Layout(self.config.lmax)
        self.edge_layout = SO2Layout(self.config.lmax, self.config.mmax)
        self.atom_embedding = nn.Embedding(
            self.config.max_atomic_number + 1, self.config.channels
        )
        self.register_buffer(
            "atomic_reference_energies",
            torch.zeros(self.config.max_atomic_number + 1),
            persistent=True,
        )
        self.register_buffer(
            "composition_energy_offsets",
            (
                torch.zeros(self.config.max_atomic_number + 1)
                if self.config.composition_energy_baseline
                == "fixed_train_residual_v1"
                else None
            ),
            persistent=True,
        )
        self.geometry = EdgeGeometry(
            self.node_layout,
            self.config.num_rbf,
            self.config.cutoff,
            self.config.min_distance,
        )
        self.blocks = nn.ModuleList(
            [
                HessSO2Block(
                    self.node_layout,
                    self.edge_layout,
                    self.config.channels,
                    self.config.num_rbf,
                    self.config.radial_hidden,
                    self.config.rank,
                    self.config.num_foci,
                    invariant_scalar_feedback=self.config.invariant_scalar_feedback,
                    invariant_scalar_feedback_epsilon=(
                        self.config.invariant_scalar_feedback_epsilon
                        if self.config.invariant_scalar_feedback
                        else 1.0e-8
                    ),
                    invariant_scalar_feedback_mode=(
                        self.config.invariant_scalar_feedback_mode
                    ),
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.energy_head = AtomicEnergyHead(self.config.channels)
        self.direct_head = (
            DirectForceHead(self.config.channels)
            if self.config.direct_force_head
            else None
        )

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "architecture": "HessSO2-Equiformer-C1-v0.1",
            "config": asdict(self.config),
            "basis_id": self.node_layout.basis_id,
            "layout_id": self.node_layout.layout_id,
            "gauge_id": "deterministic-y-two-chart-v1",
            "angular_compensation": "identity",
            "invariant_scalar_feedback": self.config.invariant_scalar_feedback,
            "invariant_scalar_feedback_mode": (
                self.config.invariant_scalar_feedback_mode
            ),
            "invariant_scalar_feedback_epsilon": float(
                self.config.invariant_scalar_feedback_epsilon
            ),
            "scalar_feedback_initialization": (
                "function_preserving_common_rng_zero_columns_v1"
                if self.config.invariant_scalar_feedback
                else "legacy_linear_v1"
            ),
            "atomic_reference_energies": {
                str(index): float(value)
                for index, value in enumerate(
                    self.atomic_reference_energies.detach().cpu().tolist()
                )
                if value != 0.0
            },
            "composition_energy_baseline": (
                self.config.composition_energy_baseline
            ),
            "composition_energy_offsets": (
                {
                    str(index): float(value)
                    for index, value in enumerate(
                        self.composition_energy_offsets.detach().cpu().tolist()
                    )
                    if value != 0.0
                }
                if self.composition_energy_offsets is not None
                else None
            ),
            "units": {
                "position": "angstrom",
                "energy": "eV",
                "force": "eV/angstrom",
                "hessian": "eV/angstrom^2",
            },
        }

    def set_atomic_reference_energies(
        self, references: dict[int, float] | Tensor
    ) -> None:
        """Set fixed per-element energy references fitted on training data only."""

        values = torch.zeros_like(self.atomic_reference_energies)
        if isinstance(references, Tensor):
            if references.shape != values.shape:
                raise ValueError(
                    "reference tensor must match the atomic embedding table"
                )
            values.copy_(references.to(device=values.device, dtype=values.dtype))
        else:
            for atomic_number, reference in references.items():
                if atomic_number <= 0 or atomic_number >= values.numel():
                    raise ValueError(
                        f"atomic reference Z={atomic_number} is outside the table"
                    )
                values[atomic_number] = float(reference)
        self.atomic_reference_energies.copy_(values)

    def set_composition_energy_offsets(
        self, offsets: dict[int, float] | Tensor
    ) -> None:
        """Set fixed coordinate-independent residual-composition offsets.

        This buffer is intentionally distinct from the theory-level atomic
        references loaded from the normalizer.  It exists only when
        ``composition_energy_baseline='fixed_train_residual_v1'`` was selected
        explicitly in the model configuration.
        """

        current = self.composition_energy_offsets
        if current is None:
            raise ValueError("composition energy baseline is disabled")
        values = torch.zeros_like(current)
        if isinstance(offsets, Tensor):
            if offsets.shape != values.shape:
                raise ValueError(
                    "composition offset tensor must match the atomic table"
                )
            values.copy_(offsets.to(device=values.device, dtype=values.dtype))
        else:
            for atomic_number, offset in offsets.items():
                if atomic_number <= 0 or atomic_number >= values.numel():
                    raise ValueError(
                        f"composition offset Z={atomic_number} is outside the table"
                    )
                values[atomic_number] = float(offset)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("composition energy offsets must be finite")
        current.copy_(values)

    def _initial_nodes(self, atomic_numbers: Tensor) -> SO3NodeTensor:
        if atomic_numbers.ndim != 1 or atomic_numbers.dtype != torch.long:
            raise TypeError("atomic_numbers must be torch.long with shape [N]")
        if atomic_numbers.numel() and int(atomic_numbers.max().item()) > self.config.max_atomic_number:
            raise ValueError("atomic number exceeds configured embedding table")
        scalar = self.atom_embedding(atomic_numbers).unsqueeze(1)
        non_scalar = scalar.new_zeros(
            (
                atomic_numbers.shape[0],
                self.node_layout.num_coefficients - 1,
                self.config.channels,
            )
        )
        return SO3NodeTensor(torch.cat((scalar, non_scalar), dim=1), self.node_layout)

    def energy_features(
        self,
        atomic_numbers: Tensor,
        pos: Tensor,
        batch: Tensor,
        topology: EdgeTopology | None = None,
        cell: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, SO3NodeTensor, EdgeTopology]:
        """Return graph energy, per-atom energy, features, and fixed topology."""

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos must have shape [N,3]")
        if batch.shape != (pos.shape[0],) or batch.dtype != torch.long:
            raise TypeError("batch must be torch.long with shape [N]")
        if atomic_numbers.shape != (pos.shape[0],):
            raise ValueError("atomic_numbers must have shape [N]")
        if topology is None:
            topology = build_complete_topology(
                batch, self.config.cutoff, self.config.skin
            )
        elif topology.src.device != pos.device:
            topology = topology.to(pos.device)
        geometry = self.geometry.materialize(pos, topology, cell)
        nodes = self._initial_nodes(atomic_numbers)
        for block in self.blocks:
            nodes, _ = block(nodes, topology, geometry)
        scalar = nodes.values[:, self.node_layout.degree_slice(0), :].squeeze(1)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        energy, per_atom = self.energy_head(scalar, batch, num_graphs)
        references = self.atomic_reference_energies.to(
            device=energy.device, dtype=energy.dtype
        ).index_select(0, atomic_numbers)
        reference_energy = energy.new_zeros((num_graphs,))
        reference_energy = reference_energy.index_add(0, batch, references)
        energy = energy + reference_energy
        per_atom = per_atom + references
        if self.composition_energy_offsets is not None:
            offsets = self.composition_energy_offsets.to(
                device=energy.device, dtype=energy.dtype
            ).index_select(0, atomic_numbers)
            composition_energy = energy.new_zeros((num_graphs,))
            composition_energy = composition_energy.index_add(0, batch, offsets)
            # Add only after the learned graph-energy reduction.  This keeps
            # the coordinate-dependent reduction graph unchanged, which is
            # required by the bitwise F/H invariance contract.
            energy = energy + composition_energy
            per_atom = per_atom + offsets
        return energy, per_atom, nodes, topology

    def forward(
        self,
        atomic_numbers: Tensor,
        pos: Tensor,
        batch: Tensor,
        topology: EdgeTopology | None = None,
        cell: Tensor | None = None,
        compute_forces: bool = True,
        create_graph: bool | None = None,
        return_features: bool = False,
    ) -> PotentialOutput:
        """Evaluate energy and explicitly selected force heads.

        If a caller supplies a coordinate tensor that already requires gradients,
        the exact tensor is preserved.  Otherwise a differentiable clone is made
        only when conservative force evaluation is requested.
        """

        coordinate_pos = pos
        if compute_forces and not coordinate_pos.requires_grad:
            coordinate_pos = coordinate_pos.clone().requires_grad_(True)
        energy, per_atom, nodes, used_topology = self.energy_features(
            atomic_numbers, coordinate_pos, batch, topology, cell
        )
        force_conservative: Tensor | None = None
        if compute_forces:
            graph_flag = self.training if create_graph is None else bool(create_graph)
            gradient = torch.autograd.grad(
                energy.sum(),
                coordinate_pos,
                create_graph=graph_flag,
                retain_graph=True,
            )[0]
            force_conservative = -gradient
        force_direct: Tensor | None = None
        if self.direct_head is not None:
            l1 = nodes.values[:, self.node_layout.degree_slice(1), :]
            force_direct = self.direct_head(l1)
        return PotentialOutput(
            energy=energy,
            force_conservative=force_conservative,
            force_direct=force_direct,
            force_mode="conservative" if compute_forces else "energy_only",
            positions=coordinate_pos,
            topology=used_topology,
            features=nodes.values if return_features else None,
            per_atom_energy=per_atom,
            provenance=self.provenance,
        )
