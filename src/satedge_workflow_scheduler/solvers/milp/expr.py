"""Linear expressions and constraints of the MILP, independent of any solver.

A :class:`LinExpr` is ``Σ coef · var + constant``, where a variable is a handle (an integer
given by the backend). The model writes its mathematics with these objects, so that a
constraint reads as in the document, for example ``x[i, s] + y <= 1``.

Products of two variables are not defined: every constraint built here is linear by
construction. A comparison (``<=``, ``>=``, ``==``) produces a :class:`Row`, which is what the
backends receive.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Self

from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError

Var = int
"""Handle of a variable, assigned by the backend when the variable is created."""


class Sense(StrEnum):
    """Relation of a row: ``expr sense rhs``."""

    LE = "<="
    GE = ">="
    EQ = "=="


class LinExpr:
    """A linear expression over variable handles, with a constant term.

    Expressions are values: the arithmetic operators return new objects. For summing many
    terms use :meth:`total`, which builds the result in one pass, or ``+=``, which updates
    the left operand in place.

    Args:
        terms: Coefficient of each variable handle. Zero coefficients are dropped.
        constant: The constant term.
    """

    __slots__ = ("_terms", "constant")

    def __init__(
        self, terms: Mapping[Var, float] | None = None, constant: float = 0.0
    ) -> None:
        self._terms: dict[Var, float] = {
            v: float(c) for v, c in (terms or {}).items() if c
        }
        self.constant = float(constant)

    @classmethod
    def var(cls, var: Var, coef: float = 1.0) -> LinExpr:
        """The expression ``coef · var``."""
        return cls({var: coef})

    @classmethod
    def number(cls, value: float) -> LinExpr:
        """The constant expression ``value``."""
        return cls(constant=value)

    @classmethod
    def total(cls, items: Iterable[LinExpr | float]) -> LinExpr:
        """The sum of ``items``, built in one pass (linear in the number of terms)."""
        result = cls()
        for item in items:
            result += item
        return result

    @property
    def terms(self) -> Mapping[Var, float]:
        """Read-only view of the coefficients of the variables."""
        return MappingProxyType(self._terms)

    def __iadd__(self, other: LinExpr | float) -> Self:
        expr = _as_expr(other)
        for var, coef in expr._terms.items():
            total = self._terms.get(var, 0.0) + coef
            if not total:
                self._terms.pop(var, None)
            else:
                self._terms[var] = total
        self.constant += expr.constant
        return self

    def __add__(self, other: LinExpr | float) -> LinExpr:
        result = LinExpr(self._terms, self.constant)
        result += other
        return result

    __radd__ = __add__

    def __neg__(self) -> LinExpr:
        return LinExpr({v: -c for v, c in self._terms.items()}, -self.constant)

    def __sub__(self, other: LinExpr | float) -> LinExpr:
        return self + (-_as_expr(other))

    def __rsub__(self, other: LinExpr | float) -> LinExpr:
        return _as_expr(other) + (-self)

    def __mul__(self, factor: float) -> LinExpr:
        if not isinstance(factor, int | float):
            return NotImplemented
        return LinExpr(
            {v: c * factor for v, c in self._terms.items()}, self.constant * factor
        )

    __rmul__ = __mul__

    def __le__(self, other: LinExpr | float) -> Row:
        return Row.compare(self, Sense.LE, other)

    def __ge__(self, other: LinExpr | float) -> Row:
        return Row.compare(self, Sense.GE, other)

    def __eq__(self, other: object) -> Row:  # type: ignore[override]
        if not isinstance(other, LinExpr | int | float):
            return NotImplemented
        return Row.compare(self, Sense.EQ, other)

    # defining __eq__ removes the default hash; an expression is a value, not a key
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        parts = [f"{c:+g}*x{v}" for v, c in self._terms.items()]
        if self.constant or not parts:
            parts.append(f"{self.constant:+g}")
        return "LinExpr(" + " ".join(parts) + ")"


def _as_expr(item: LinExpr | float) -> LinExpr:
    if isinstance(item, LinExpr):
        return item
    return LinExpr.number(item)


@dataclass(frozen=True)
class Row:
    """A linear constraint ``Σ coef · var  sense  rhs``, with the constant moved to ``rhs``.

    Attributes:
        terms: Coefficient of each variable handle, with no constant term.
        sense: Relation of the row.
        rhs: Right-hand side.
        name: Optional name, kept for debugging and for the statistics of the blocks.
    """

    terms: Mapping[Var, float]
    sense: Sense
    rhs: float
    name: str = ""

    @classmethod
    def compare(
        cls, lhs: LinExpr, sense: Sense, rhs: LinExpr | float, name: str = ""
    ) -> Row:
        """The row ``lhs sense rhs``: both sides are moved to the left, then the constants
        are moved back to the right."""
        diff = lhs - _as_expr(rhs)
        return cls(dict(diff.terms), sense, -diff.constant, name)

    def named(self, name: str) -> Row:
        """The same row with a new name."""
        return Row(self.terms, self.sense, self.rhs, name)


_TOLERANCE = 1e-9


def screen_empty_rows(rows: Iterable[Row]) -> Iterator[Row]:
    """The rows that still contain a variable, in order.

    A row with no variable left, for example ``0 ≤ 1 − 0`` once the zero terms have disappeared,
    is a constant. If it holds it is dropped: it constrains nothing. If it does not hold, the
    model is infeasible by construction, and the error says which row it is.

    Raises:
        ModelBuildError: If a row without variables is never satisfied.
    """
    for row in rows:
        if row.terms:
            yield row
            continue
        lhs, rhs = 0.0, row.rhs
        holds = {
            Sense.LE: lhs <= rhs + _TOLERANCE,
            Sense.GE: lhs >= rhs - _TOLERANCE,
            Sense.EQ: abs(lhs - rhs) <= _TOLERANCE,
        }[row.sense]
        if not holds:
            raise ModelBuildError(
                f"row {row.name!r} has no variable and is never satisfied"
            )
