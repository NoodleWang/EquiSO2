"""Immutable discrete edge topology for molecular batches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import Tensor


def _topology_digest(src: Tensor, dst: Tensor, rcut: float, skin: float) -> str:
    payload = (
        src.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy().tobytes()
        + dst.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy().tobytes()
        + f"|{rcut:.12g}|{skin:.12g}".encode("ascii")
    )
    return hashlib.sha1(payload).hexdigest()[:16]


@dataclass(frozen=True)
class EdgeTopology:
    """Coordinate-independent directed edges.

    ``src`` is the sending atom ``j`` and ``dst`` is the receiving atom ``i``;
    geometry therefore uses ``r_ij = R_j - R_i``.  No edge vector is stored.
    Shapes follow ``whf.md``, Section 8.2.
    """

    src: Tensor
    dst: Tensor
    cell_shift: Tensor
    edge_batch: Tensor
    rcut: float
    skin: float
    topology_id: str

    def __post_init__(self) -> None:
        edge_count = int(self.src.numel())
        if self.src.ndim != 1 or self.dst.shape != self.src.shape:
            raise ValueError("src and dst must both have shape [E]")
        if self.src.dtype != torch.long or self.dst.dtype != torch.long:
            raise TypeError("src and dst must use torch.long")
        if self.cell_shift.shape != (edge_count, 3):
            raise ValueError("cell_shift must have shape [E, 3]")
        if self.edge_batch.shape != (edge_count,):
            raise ValueError("edge_batch must have shape [E]")
        if self.rcut <= 0.0 or self.skin < 0.0:
            raise ValueError("rcut must be positive and skin non-negative")

    @property
    def edge_index(self) -> Tensor:
        return torch.stack((self.src, self.dst), dim=0)

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())

    def to(self, device: torch.device | str) -> "EdgeTopology":
        return EdgeTopology(
            src=self.src.to(device=device),
            dst=self.dst.to(device=device),
            cell_shift=self.cell_shift.to(device=device),
            edge_batch=self.edge_batch.to(device=device),
            rcut=self.rcut,
            skin=self.skin,
            topology_id=self.topology_id,
        )


def _make_topology(
    src: Tensor,
    dst: Tensor,
    batch: Tensor,
    rcut: float,
    skin: float,
) -> EdgeTopology:
    edge_batch = batch.index_select(0, dst)
    shifts = torch.zeros((src.numel(), 3), dtype=torch.long, device=src.device)
    return EdgeTopology(
        src=src,
        dst=dst,
        cell_shift=shifts,
        edge_batch=edge_batch,
        rcut=float(rcut),
        skin=float(skin),
        topology_id=_topology_digest(src, dst, rcut, skin),
    )


def build_complete_topology(
    batch: Tensor,
    rcut: float,
    skin: float = 0.0,
) -> EdgeTopology:
    """Build all directed non-self edges within each molecule.

    The complete candidate graph is the default fixed-topology contract for the
    supplied small molecules (ambiguity log A8).  Physical interactions still
    vanish exactly at ``rcut`` through the C3 envelope.
    """

    if batch.ndim != 1 or batch.dtype != torch.long:
        raise TypeError("batch must be a torch.long tensor with shape [N]")
    same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
    non_self = ~torch.eye(batch.numel(), dtype=torch.bool, device=batch.device)
    # Matrix axes are [dst, src], matching message aggregation semantics.
    dst, src = torch.nonzero(same_graph & non_self, as_tuple=True)
    return _make_topology(src, dst, batch, rcut, skin)


def build_radius_topology(
    pos: Tensor,
    batch: Tensor,
    rcut: float,
    skin: float,
) -> EdgeTopology:
    """Build a discrete ``rcut + skin`` candidate list at reference positions.

    This operation is intentionally discrete.  The returned topology may be
    reused for finite-difference coordinates, but geometry must be rematerialized
    on every call.  Use :func:`build_complete_topology` when the skin coverage
    cannot be proven.
    """

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape [N, 3]")
    if batch.shape != (pos.shape[0],):
        raise ValueError("batch must have shape [N]")
    candidate = float(rcut) + float(skin)
    if candidate <= 0.0:
        raise ValueError("rcut + skin must be positive")
    delta = pos.detach().unsqueeze(0) - pos.detach().unsqueeze(1)
    distance = torch.linalg.vector_norm(delta, dim=-1)
    same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
    non_self = ~torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
    dst, src = torch.nonzero(
        same_graph & non_self & (distance <= candidate), as_tuple=True
    )
    return _make_topology(src, dst, batch, rcut, skin)

