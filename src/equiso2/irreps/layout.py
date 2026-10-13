"""One explicit coefficient ordering for every equivariant tensor."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


BASIS_ID = "e3nn-real-y-v1"
LAYOUT_ID = "degree-major-m-ascending-v1"


@dataclass(frozen=True)
class SO3Layout:
    """Full global SO(3) layout with degree-major e3nn coefficients."""

    lmax: int
    basis_id: str = BASIS_ID
    layout_id: str = LAYOUT_ID

    def __post_init__(self) -> None:
        if self.lmax < 0:
            raise ValueError("lmax must be non-negative")

    @property
    def num_coefficients(self) -> int:
        return (self.lmax + 1) ** 2

    def degree_slice(self, degree: int) -> slice:
        if degree < 0 or degree > self.lmax:
            raise IndexError(f"degree {degree} outside 0..{self.lmax}")
        return slice(degree * degree, (degree + 1) * (degree + 1))

    @staticmethod
    def degree_m_index(degree: int, order: int) -> int:
        if abs(order) > degree:
            raise IndexError(f"|m|={abs(order)} exceeds l={degree}")
        return degree * degree + order + degree


@dataclass(frozen=True)
class SO2Layout:
    """Compact edge-local ``|m| <= mmax`` layout.

    Coefficients are degree-major, then ascending ``m`` within each retained
    degree.  The count is ``(M+1)^2 + (L-M)(2M+1)`` from ``whf.md``, Section 4.4.
    """

    lmax: int
    mmax: int
    basis_id: str = BASIS_ID
    layout_id: str = LAYOUT_ID

    def __post_init__(self) -> None:
        if self.lmax < 0:
            raise ValueError("lmax must be non-negative")
        if self.mmax < 0 or self.mmax > self.lmax:
            raise ValueError("mmax must satisfy 0 <= mmax <= lmax")

    @property
    def entries(self) -> tuple[tuple[int, int], ...]:
        values: list[tuple[int, int]] = []
        for degree in range(self.lmax + 1):
            bound = min(degree, self.mmax)
            for order in range(-bound, bound + 1):
                values.append((degree, order))
        return tuple(values)

    @property
    def num_coefficients(self) -> int:
        return (self.mmax + 1) ** 2 + (self.lmax - self.mmax) * (
            2 * self.mmax + 1
        )

    def compact_index(self, degree: int, order: int) -> int:
        try:
            return self.entries.index((degree, order))
        except ValueError as error:
            raise IndexError(
                f"coefficient (l={degree}, m={order}) is not in {self}"
            ) from error

    def degree_indices(self, degree: int) -> tuple[int, ...]:
        return tuple(
            index for index, (ell, _) in enumerate(self.entries) if ell == degree
        )

    def signed_order_indices(self, abs_order: int, sign: int = 1) -> tuple[int, ...]:
        """Indices ordered by degree for one signed m component."""

        if abs_order < 0 or abs_order > self.mmax:
            raise IndexError(f"|m|={abs_order} outside 0..{self.mmax}")
        if abs_order == 0:
            order = 0
        else:
            if sign not in (-1, 1):
                raise ValueError("sign must be -1 or +1")
            order = sign * abs_order
        return tuple(
            self.compact_index(degree, order)
            for degree in range(abs_order, self.lmax + 1)
        )


@dataclass(frozen=True)
class SO3NodeTensor:
    """Global node irreps, values shaped ``[N,(L+1)^2,C]``."""

    values: Tensor
    layout: SO3Layout

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("SO3NodeTensor values must have shape [N,D,C]")
        if self.values.shape[1] != self.layout.num_coefficients:
            raise ValueError("coefficient dimension does not match SO3 layout")

    @property
    def num_nodes(self) -> int:
        return int(self.values.shape[0])

    @property
    def channels(self) -> int:
        return int(self.values.shape[2])


@dataclass(frozen=True)
class SO2EdgeTensor:
    """Edge-local coefficients tied to one topology and gauge."""

    values: Tensor
    layout: SO2Layout
    topology_id: str
    gauge_id: str

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("SO2EdgeTensor values must have shape [E,D,C]")
        if self.values.shape[1] != self.layout.num_coefficients:
            raise ValueError("coefficient dimension does not match SO2 layout")

    @property
    def num_edges(self) -> int:
        return int(self.values.shape[0])

    @property
    def channels(self) -> int:
        return int(self.values.shape[2])

