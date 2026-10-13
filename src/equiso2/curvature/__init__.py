"""Curvature operators with explicit Jacobian orientation semantics.

The project uses three deliberately distinct names:

``minus_Jv``
    :math:`-J_F v`, obtained with a JVP or a centered force difference.
``minus_JT_v``
    :math:`-J_F^T v`, obtained with a VJP.
``Hv``
    :math:`H_E v`, used only for an energy Hessian or when force
    conservativity/symmetry is an explicit assumption.

Keeping the names separate prevents a direct-force Jacobian from silently being
reported as a physical Hessian (``docs/whf.md``, sections 5.1--5.4 and 8.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor


ActionName = Literal["minus_Jv", "minus_JT_v", "Hv"]


@dataclass(frozen=True)
class CurvatureResult:
    """Result of applying a curvature/Jacobian operator to one or more probes.

    Parameters
    ----------
    method
        Human-readable backend name, for example ``"direct_vjp"`` or
        ``"conservative_fd_hvp"``.
    probe
        Probe tensor. A single probe has shape ``[D]`` and multiple probes have
        shape ``[P, D]``, where ``D = 3 * n_atoms``.
    minus_Jv, minus_JT_v, Hv
        Explicitly oriented actions. Any unavailable action is ``None``.
    step_size
        Finite-difference step metadata. Canonical FD routines store a ``[P,B]``
        tensor so per-structure steps remain auditable.
    ptr
        Optional atom pointer with shape ``[B+1]``.
    topology_id
        Opaque identifier for the fixed discrete topology used by the call.
    valid_mask
        Optional probe/structure validity mask, normally shape ``[P,B]``.
    """

    method: str
    probe: Tensor
    minus_Jv: Tensor | None = None
    minus_JT_v: Tensor | None = None
    Hv: Tensor | None = None
    step_size: float | Tensor | None = None
    ptr: Tensor | None = None
    topology_id: Any | None = None
    valid_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.probe, Tensor):
            raise TypeError("`probe` must be a torch.Tensor")
        actions = (self.minus_Jv, self.minus_JT_v, self.Hv)
        if all(action is None for action in actions):
            raise ValueError("A CurvatureResult must contain at least one action")
        for name, action in zip(
            ("minus_Jv", "minus_JT_v", "Hv"), actions, strict=True
        ):
            if action is not None and action.shape != self.probe.shape:
                raise ValueError(
                    f"`{name}` shape {tuple(action.shape)} does not match probe "
                    f"shape {tuple(self.probe.shape)}"
                )

    @property
    def available_actions(self) -> tuple[str, ...]:
        """Return the explicitly available action names."""

        return tuple(
            name
            for name in ("minus_Jv", "minus_JT_v", "Hv")
            if getattr(self, name) is not None
        )

    def action(self, name: ActionName) -> Tensor:
        """Return one action, failing rather than changing orientation silently."""

        value = getattr(self, name)
        if value is None:
            available = ", ".join(self.available_actions)
            raise ValueError(
                f"Curvature action {name!r} is unavailable; available: {available}"
            )
        return value


# Imports intentionally follow CurvatureResult: operator modules import the
# dataclass from this module without creating a circular initialization failure.
from .finite_difference import (  # noqa: E402
    finite_difference_force_jvp,
    finite_difference_hvp,
)
from .hvp import exact_energy_hvp, minus_force_jacobian_vector  # noqa: E402
from .sampler import (  # noqa: E402
    ProbeBatch,
    sample_coordinate_probes,
    sample_isotropic_probes,
    sample_probes,
    validate_ptr,
)
from .vjp import (  # noqa: E402
    direct_force_vjp,
    minus_force_jacobian_transpose_vector,
)

__all__ = [
    "ActionName",
    "CurvatureResult",
    "ProbeBatch",
    "direct_force_vjp",
    "exact_energy_hvp",
    "finite_difference_force_jvp",
    "finite_difference_hvp",
    "minus_force_jacobian_transpose_vector",
    "minus_force_jacobian_vector",
    "sample_coordinate_probes",
    "sample_isotropic_probes",
    "sample_probes",
    "validate_ptr",
]
