"""Typed model output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from equiso2.graph.topology import EdgeTopology


@dataclass
class PotentialOutput:
    """Potential output with force semantics made explicit.

    ``force_conservative`` is always ``-grad_pos(sum(energy))``.  A direct force
    is an auxiliary output and is never silently added to it (``whf.md``,
    Sections 4.8 and 8.2).
    """

    energy: Tensor
    force_conservative: Tensor | None
    force_direct: Tensor | None
    force_mode: str
    positions: Tensor
    topology: EdgeTopology
    features: Tensor | None = None
    per_atom_energy: Tensor | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

