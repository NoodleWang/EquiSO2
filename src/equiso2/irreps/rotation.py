"""Explicit global-to-local projection and local-to-global lifting."""

from __future__ import annotations

import torch
from torch import Tensor

from equiso2.graph.geometry import GeometryCache
from equiso2.graph.topology import EdgeTopology

from .layout import SO2EdgeTensor, SO2Layout, SO3NodeTensor


def project_to_edge_gauge(
    nodes: SO3NodeTensor,
    topology: EdgeTopology,
    geometry: GeometryCache,
    edge_layout: SO2Layout,
) -> SO2EdgeTensor:
    """Rotate source nodes into each edge gauge and truncate ``|m|``.

    Input values are ``[N,(L+1)^2,C]`` and output values are
    ``[E,D_{L,M},C]``.  Wigner matrices remain connected to coordinates.
    This is the explicit conversion required by ``whf.md``, Section 8.2.
    """

    if nodes.layout.lmax != edge_layout.lmax:
        raise ValueError("node and edge lmax must match")
    if nodes.layout.basis_id != edge_layout.basis_id:
        raise ValueError("node and edge bases must match")
    if topology.topology_id != geometry.topology_id:
        raise ValueError("topology and geometry ids do not match")
    src = topology.src.to(device=nodes.values.device)
    source_values = nodes.values.index_select(0, src)
    pieces: list[Tensor] = []
    for degree in range(nodes.layout.lmax + 1):
        degree_values = source_values[:, nodes.layout.degree_slice(degree), :]
        wigner = geometry.wigner[degree]
        local = torch.einsum("eji,ejc->eic", wigner, degree_values)
        bound = min(degree, edge_layout.mmax)
        keep = torch.arange(
            degree - bound,
            degree + bound + 1,
            dtype=torch.long,
            device=local.device,
        )
        pieces.append(local.index_select(1, keep))
    values = torch.cat(pieces, dim=1)
    return SO2EdgeTensor(
        values=values,
        layout=edge_layout,
        topology_id=topology.topology_id,
        gauge_id=geometry.gauge_id,
    )


def lift_to_global(edges: SO2EdgeTensor, geometry: GeometryCache) -> Tensor:
    """Zero-pad local coefficients and rotate them back to global irreps.

    Returns edge messages with shape ``[E,(L+1)^2,C]``.  The identity angular
    compensation choice is documented in ambiguity log A5.
    """

    if edges.topology_id != geometry.topology_id:
        raise ValueError("edge tensor and geometry topology ids do not match")
    if edges.gauge_id != geometry.gauge_id:
        raise ValueError("edge tensor and geometry gauge ids do not match")
    pieces: list[Tensor] = []
    compact_offset = 0
    for degree in range(edges.layout.lmax + 1):
        bound = min(degree, edges.layout.mmax)
        retained = 2 * bound + 1
        compact = edges.values[:, compact_offset : compact_offset + retained, :]
        compact_offset += retained
        local = edges.values.new_zeros(
            (edges.num_edges, 2 * degree + 1, edges.channels)
        )
        keep = torch.arange(
            degree - bound,
            degree + bound + 1,
            dtype=torch.long,
            device=edges.values.device,
        )
        local = torch.index_copy(local, 1, keep, compact)
        wigner = geometry.wigner[degree]
        global_degree = torch.einsum("eij,ejc->eic", wigner, local)
        pieces.append(global_degree)
    return torch.cat(pieces, dim=1)


def aggregate_global_messages(
    edge_values: Tensor, dst: Tensor, num_nodes: int
) -> Tensor:
    """Sum global edge messages at destinations without PyG extensions."""

    if edge_values.ndim != 3:
        raise ValueError("edge_values must have shape [E,D,C]")
    if dst.shape != (edge_values.shape[0],):
        raise ValueError("dst must have shape [E]")
    output = edge_values.new_zeros(
        (num_nodes, edge_values.shape[1], edge_values.shape[2])
    )
    return output.index_add(0, dst.to(device=edge_values.device), edge_values)

