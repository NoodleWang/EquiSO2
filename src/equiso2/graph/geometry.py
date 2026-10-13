"""Pure, coordinate-differentiable edge geometry materialization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import TYPE_CHECKING

import torch
from e3nn import o3
from torch import Tensor, nn

from .cutoff import GaussianRadialBasis, c3_envelope
from .topology import EdgeTopology

if TYPE_CHECKING:
    from equiso2.irreps.layout import SO3Layout


GAUGE_ID = "deterministic-y-two-chart-v1"


def deterministic_edge_frame(unit: Tensor) -> Tensor:
    """Construct a deterministic right-handed frame with local y along an edge.

    Args:
        unit: Strict unit directions with shape ``[E, 3]``.

    Returns:
        Local-to-global rotation matrices with shape ``[E, 3, 3]``.  Columns are
        local x, y, z axes.  The implementation uses two transverse-axis charts
        (ambiguity log A3); all SO(2) operators must be independent of their roll.

    No epsilon is added to a valid normalization denominator, as required by
    ``whf.md``, Section 4.2.
    """

    if unit.ndim != 2 or unit.shape[1] != 3:
        raise ValueError(f"unit must have shape [E, 3], got {unit.shape}")
    ref_y = unit.new_tensor((0.0, 1.0, 0.0)).unsqueeze(0).expand_as(unit)
    ref_z = unit.new_tensor((0.0, 0.0, 1.0)).unsqueeze(0).expand_as(unit)
    use_y = (unit[:, 1].abs() < 0.9).unsqueeze(1)
    reference = torch.where(use_y, ref_y, ref_z)
    axis_x_raw = torch.linalg.cross(reference, unit, dim=-1)
    axis_x_norm = torch.linalg.vector_norm(axis_x_raw, dim=-1, keepdim=True)
    axis_x = axis_x_raw / axis_x_norm
    axis_z = torch.linalg.cross(axis_x, unit, dim=-1)
    return torch.stack((axis_x, unit, axis_z), dim=-1)


@lru_cache(maxsize=None)
def _harmonic_sampling_basis(degree: int) -> tuple[Tensor, Tensor]:
    """Return deterministic CPU/float64 samples and their harmonic inverse."""

    dimension = 2 * degree + 1
    count = max(2 * dimension, 8)
    index = torch.arange(count, dtype=torch.float64)
    height = 1.0 - 2.0 * (index + 0.5) / count
    angle = index * (math.pi * (3.0 - math.sqrt(5.0)))
    radius = torch.sqrt(torch.clamp(1.0 - height.square(), min=0.0))
    directions = torch.stack(
        (radius * torch.cos(angle), height, radius * torch.sin(angle)), dim=-1
    )
    harmonics = o3.spherical_harmonics(
        degree,
        directions,
        normalize=True,
        normalization="component",
    )
    return directions, torch.linalg.pinv(harmonics)


def wigner_from_frame(frame: Tensor, lmax: int) -> tuple[Tensor, ...]:
    """Return smooth real Wigner matrices directly from rotation matrices.

    Euler extraction differentiates through ``acos`` and is singular for an
    edge exactly parallel to the global y axis.  Equivariance of e3nn's real
    spherical harmonics recovers the same representation matrices from a fixed
    overdetermined set of directions without Euler coordinates.
    """

    if frame.ndim != 3 or frame.shape[1:] != (3, 3):
        raise ValueError("frame must have shape [E, 3, 3]")
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    result: list[Tensor] = []
    for degree in range(lmax + 1):
        if degree == 0:
            result.append(frame.new_ones((frame.shape[0], 1, 1)))
        elif degree == 1:
            # e3nn's real l=1 representation uses the Cartesian xyz basis.
            result.append(frame)
        else:
            directions_cpu, inverse_cpu = _harmonic_sampling_basis(degree)
            directions = directions_cpu.to(
                device=frame.device, dtype=frame.dtype
            )
            inverse = inverse_cpu.to(device=frame.device, dtype=frame.dtype)
            rotated = torch.einsum("eij,kj->eki", frame, directions)
            harmonics = o3.spherical_harmonics(
                degree,
                rotated,
                normalize=True,
                normalization="component",
            )
            representation_transpose = torch.einsum(
                "dk,ekf->edf", inverse, harmonics
            )
            result.append(representation_transpose.transpose(-1, -2))
    return tuple(result)


@dataclass(frozen=True)
class GeometryCache:
    """Immutable geometry shared only inside one forward call."""

    vectors: Tensor
    distances: Tensor
    unit: Tensor
    frame: Tensor
    wigner: tuple[Tensor, ...]
    radial: Tensor
    envelope: Tensor
    topology_id: str
    gauge_id: str
    requires_coordinate_grad: bool


class EdgeGeometry(nn.Module):
    """Materialize all continuous edge quantities from current coordinates."""

    def __init__(
        self,
        layout: "SO3Layout",
        num_rbf: int,
        cutoff: float,
        min_distance: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if min_distance <= 0.0:
            raise ValueError("min_distance must be positive")
        self.layout = layout
        self.cutoff = float(cutoff)
        self.min_distance = float(min_distance)
        self.rbf = GaussianRadialBasis(num_rbf, cutoff)

    def materialize(
        self,
        pos: Tensor,
        topology: EdgeTopology,
        cell: Tensor | None = None,
    ) -> GeometryCache:
        """Recompute vec/dist/frame/Wigner/RBF for one coordinate tensor.

        ``pos`` has shape ``[N,3]``.  A molecular ``cell`` may be ``[3,3]`` or a
        batched ``[B,3,3]`` tensor.  No coordinate-derived tensor is detached or
        written to module state (``whf.md``, Sections 4.2 and 8.3).
        """

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos must have shape [N,3], got {pos.shape}")
        src = topology.src.to(device=pos.device)
        dst = topology.dst.to(device=pos.device)
        vectors = pos.index_select(0, src) - pos.index_select(0, dst)
        if cell is not None:
            shifts = topology.cell_shift.to(device=pos.device, dtype=pos.dtype)
            if cell.ndim == 2:
                vectors = vectors + torch.einsum("ei,ij->ej", shifts, cell)
            elif cell.ndim == 3:
                edge_batch = topology.edge_batch.to(device=pos.device)
                edge_cell = cell.index_select(0, edge_batch)
                vectors = vectors + torch.einsum("ei,eij->ej", shifts, edge_cell)
            else:
                raise ValueError("cell must have shape [3,3] or [B,3,3]")
        distances = torch.linalg.vector_norm(vectors, dim=-1)
        if distances.numel() and bool((distances < self.min_distance).any().item()):
            bad = float(distances.min().detach().cpu())
            raise ValueError(
                f"overlapping atoms/invalid edge: minimum distance {bad:.3e} "
                f"is below {self.min_distance:.3e}"
            )
        unit = vectors / distances.unsqueeze(1)
        frame = deterministic_edge_frame(unit)
        wigner = wigner_from_frame(frame, self.layout.lmax)
        envelope = c3_envelope(distances, self.cutoff)
        radial = self.rbf(distances)
        return GeometryCache(
            vectors=vectors,
            distances=distances,
            unit=unit,
            frame=frame,
            wigner=wigner,
            radial=radial,
            envelope=envelope,
            topology_id=topology.topology_id,
            gauge_id=GAUGE_ID,
            requires_coordinate_grad=bool(pos.requires_grad),
        )
