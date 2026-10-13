"""Random curvature probes for variable-size atom batches.

Each row of :class:`ProbeBatch` contains one independent probe per eligible
structure. Coordinate slices are defined by ``3 * ptr``. Consequently every
active structure has a unit-norm probe and satisfies
:math:`E[v_b v_b^T] = I/(3N_b)` independently of the other structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor


ProbeKind = Literal["coordinate", "isotropic"]


def validate_ptr(ptr: Tensor | Sequence[int], *, n_atoms: int | None = None) -> Tensor:
    """Validate and return an atom pointer tensor with shape ``[B+1]``."""

    ptr_t = torch.as_tensor(ptr, dtype=torch.long)
    if ptr_t.ndim != 1 or ptr_t.numel() < 2:
        raise ValueError("`ptr` must be one-dimensional with at least two entries")
    if int(ptr_t[0].item()) != 0:
        raise ValueError("`ptr` must start at zero")
    if torch.any(ptr_t[1:] <= ptr_t[:-1]):
        raise ValueError("Every structure must contain at least one atom")
    if n_atoms is not None and int(ptr_t[-1].item()) != int(n_atoms):
        raise ValueError(
            f"`ptr[-1]` ({int(ptr_t[-1])}) does not match n_atoms ({n_atoms})"
        )
    return ptr_t


@dataclass(frozen=True)
class ProbeBatch:
    """Ragged-batch curvature probes in a flat Cartesian layout.

    ``values`` has shape ``[P,D]``, ``structure_mask`` has shape ``[P,B]``,
    and ``coordinate_indices`` optionally stores local coordinate ids in
    ``[0, 3*N_b)`` (``-1`` for inactive or non-coordinate probes).
    """

    values: Tensor
    ptr: Tensor
    structure_mask: Tensor
    kind: ProbeKind
    coordinate_indices: Tensor | None = None

    def __post_init__(self) -> None:
        ptr = validate_ptr(self.ptr)
        if self.values.ndim != 2:
            raise ValueError("`values` must have shape [n_probe, n_coordinate]")
        n_structure = ptr.numel() - 1
        expected_dof = 3 * int(ptr[-1].item())
        if self.values.shape[1] != expected_dof:
            raise ValueError(
                f"Probe width {self.values.shape[1]} does not match 3*ptr[-1] "
                f"({expected_dof})"
            )
        if self.structure_mask.shape != (self.values.shape[0], n_structure):
            raise ValueError("`structure_mask` must have shape [n_probe, batch]")
        if self.structure_mask.dtype is not torch.bool:
            raise TypeError("`structure_mask` must be boolean")
        if self.coordinate_indices is not None and self.coordinate_indices.shape != (
            self.values.shape[0],
            n_structure,
        ):
            raise ValueError("`coordinate_indices` must have shape [n_probe, batch]")

    @property
    def num_probes(self) -> int:
        return self.values.shape[0]

    @property
    def num_structures(self) -> int:
        return self.ptr.numel() - 1

    @property
    def num_probes_per_structure(self) -> Tensor:
        return self.structure_mask.sum(dim=0)


def _probe_counts(
    num_probes: int | Sequence[int] | Tensor,
    n_structure: int,
    *,
    device: torch.device,
) -> Tensor:
    counts = torch.as_tensor(num_probes, dtype=torch.long, device=device)
    if counts.ndim == 0:
        counts = counts.expand(n_structure).clone()
    elif counts.ndim != 1 or counts.numel() != n_structure:
        raise ValueError("`num_probes` must be an int or contain one count per structure")
    if torch.any(counts < 0):
        raise ValueError("Probe counts must be non-negative")
    return counts


def _valid_structures(
    valid_mask: Tensor | Sequence[bool] | None,
    n_structure: int,
    *,
    device: torch.device,
) -> Tensor:
    if valid_mask is None:
        return torch.ones(n_structure, dtype=torch.bool, device=device)
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
    if mask.shape != (n_structure,):
        raise ValueError("`valid_mask` must contain one boolean per structure")
    return mask


def _empty_probe_batch(
    ptr: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    kind: ProbeKind,
    coordinate: bool,
) -> ProbeBatch:
    n_structure = ptr.numel() - 1
    values = torch.zeros((0, 3 * int(ptr[-1])), dtype=dtype, device=device)
    mask = torch.zeros((0, n_structure), dtype=torch.bool, device=device)
    indices = (
        torch.empty((0, n_structure), dtype=torch.long, device=device)
        if coordinate
        else None
    )
    return ProbeBatch(values, ptr.to(device), mask, kind, indices)


def _sampling_device(
    target_device: torch.device, generator: torch.Generator | None
) -> torch.device:
    """Use the generator's device, then transfer samples if necessary."""

    if generator is None:
        return target_device
    return torch.device(generator.device)


def sample_coordinate_probes(
    ptr: Tensor | Sequence[int],
    num_probes: int | Sequence[int] | Tensor = 1,
    *,
    valid_mask: Tensor | Sequence[bool] | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.get_default_dtype(),
    device: torch.device | str | None = None,
) -> ProbeBatch:
    """Sample uniformly random unit coordinate probes per structure.

    This is the coordinate distribution from ``docs/whf.md`` section 5.1. A
    structure with zero requested probes or ``valid_mask=False`` contributes no
    probe and must therefore not enter a curvature-loss denominator.
    """

    ptr_cpu = validate_ptr(ptr)
    target_device = torch.device(device) if device is not None else ptr_cpu.device
    ptr_t = ptr_cpu.to(target_device)
    n_structure = ptr_t.numel() - 1
    counts = _probe_counts(num_probes, n_structure, device=target_device)
    valid = _valid_structures(valid_mask, n_structure, device=target_device)
    counts = torch.where(valid, counts, torch.zeros_like(counts))
    max_count = int(counts.max().item()) if counts.numel() else 0
    if max_count == 0:
        return _empty_probe_batch(
            ptr_t,
            dtype=dtype,
            device=target_device,
            kind="coordinate",
            coordinate=True,
        )

    total_dof = 3 * int(ptr_t[-1].item())
    values = torch.zeros((max_count, total_dof), dtype=dtype, device=target_device)
    mask = torch.zeros(
        (max_count, n_structure), dtype=torch.bool, device=target_device
    )
    indices = torch.full(
        (max_count, n_structure), -1, dtype=torch.long, device=target_device
    )
    coordinate_ptr = 3 * ptr_t
    random_device = _sampling_device(target_device, generator)
    for b in range(n_structure):
        count = int(counts[b].item())
        if count == 0:
            continue
        width = int((coordinate_ptr[b + 1] - coordinate_ptr[b]).item())
        local = torch.randint(
            width,
            (count,),
            generator=generator,
            device=random_device,
        ).to(target_device)
        rows = torch.arange(count, device=target_device)
        values[rows, coordinate_ptr[b] + local] = 1.0
        mask[:count, b] = True
        indices[:count, b] = local
    return ProbeBatch(values, ptr_t, mask, "coordinate", indices)


def sample_isotropic_probes(
    ptr: Tensor | Sequence[int],
    num_probes: int | Sequence[int] | Tensor = 1,
    *,
    valid_mask: Tensor | Sequence[bool] | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.get_default_dtype(),
    device: torch.device | str | None = None,
) -> ProbeBatch:
    """Sample unit-sphere probes independently within every structure block."""

    ptr_cpu = validate_ptr(ptr)
    target_device = torch.device(device) if device is not None else ptr_cpu.device
    ptr_t = ptr_cpu.to(target_device)
    n_structure = ptr_t.numel() - 1
    counts = _probe_counts(num_probes, n_structure, device=target_device)
    valid = _valid_structures(valid_mask, n_structure, device=target_device)
    counts = torch.where(valid, counts, torch.zeros_like(counts))
    max_count = int(counts.max().item()) if counts.numel() else 0
    if max_count == 0:
        return _empty_probe_batch(
            ptr_t,
            dtype=dtype,
            device=target_device,
            kind="isotropic",
            coordinate=False,
        )

    total_dof = 3 * int(ptr_t[-1].item())
    values = torch.zeros((max_count, total_dof), dtype=dtype, device=target_device)
    mask = torch.zeros(
        (max_count, n_structure), dtype=torch.bool, device=target_device
    )
    coordinate_ptr = 3 * ptr_t
    random_device = _sampling_device(target_device, generator)
    for b in range(n_structure):
        count = int(counts[b].item())
        if count == 0:
            continue
        start = int(coordinate_ptr[b].item())
        stop = int(coordinate_ptr[b + 1].item())
        block = torch.randn(
            (count, stop - start),
            generator=generator,
            dtype=dtype,
            device=random_device,
        ).to(target_device)
        norms = torch.linalg.vector_norm(block, dim=1, keepdim=True)
        if torch.any(norms == 0):  # practically impossible, but never divide silently
            raise RuntimeError("A sampled isotropic probe had zero norm")
        values[:count, start:stop] = block / norms
        mask[:count, b] = True
    return ProbeBatch(values, ptr_t, mask, "isotropic")


def sample_probes(
    ptr: Tensor | Sequence[int],
    num_probes: int | Sequence[int] | Tensor = 1,
    *,
    kind: ProbeKind = "coordinate",
    valid_mask: Tensor | Sequence[bool] | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.get_default_dtype(),
    device: torch.device | str | None = None,
) -> ProbeBatch:
    """Dispatch to the requested probe distribution."""

    if kind == "coordinate":
        return sample_coordinate_probes(
            ptr,
            num_probes,
            valid_mask=valid_mask,
            generator=generator,
            dtype=dtype,
            device=device,
        )
    if kind == "isotropic":
        return sample_isotropic_probes(
            ptr,
            num_probes,
            valid_mask=valid_mask,
            generator=generator,
            dtype=dtype,
            device=device,
        )
    raise ValueError(f"Unknown probe kind: {kind!r}")


def _probe_values(
    probes: Tensor | ProbeBatch,
    *,
    expected_dof: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, bool, Tensor | None, Tensor | None]:
    """Canonicalize probes for curvature backends (private shared helper)."""

    if isinstance(probes, ProbeBatch):
        values = probes.values.to(device=device, dtype=dtype)
        if values.shape[1] != expected_dof:
            raise ValueError("Probe width does not match the position tensor")
        return values, False, probes.ptr.to(device), probes.structure_mask.to(device)
    values = torch.as_tensor(probes, dtype=dtype, device=device)
    squeeze = values.ndim == 1
    if squeeze:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[1] != expected_dof:
        raise ValueError(f"Probes must have shape [{expected_dof}] or [P,{expected_dof}]")
    return values, squeeze, None, None


def _restore_probe_axis(value: Tensor, squeeze: bool) -> Tensor:
    return value.squeeze(0) if squeeze else value


__all__ = [
    "ProbeBatch",
    "ProbeKind",
    "sample_coordinate_probes",
    "sample_isotropic_probes",
    "sample_probes",
    "validate_ptr",
]
