"""In-memory backend that records the model instead of solving it.

It implements :class:`ModelSink` and :class:`MilpBackend`, so the constraint blocks and the
controller can be tested without a solver library and without a licence. Its tests check what
the model contains: the variables, their bounds, the rows with their coefficients, and the
objective. Its solve returns a result the test chooses, so that the code that reads the values
can be tested too.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from satedge_workflow_scheduler.solvers.milp.backends.ports import (
    BackendStatus,
    SolveParams,
    SolveResult,
    VarKind,
)
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import Row, Var


@dataclass(frozen=True)
class RecordedVar:
    """A variable as the model declared it.

    Attributes:
        handle: The handle returned to the model.
        lb: Lower bound.
        ub: Upper bound.
        kind: Domain of the variable.
        name: Name given by the model, possibly empty.
    """

    handle: Var
    lb: float
    ub: float
    kind: VarKind
    name: str


@dataclass
class FakeRecordingBackend:
    """Records the variables, the rows and the objective of a model.

    Attributes:
        variables: The variables, in the order they were added.
        rows: The rows, in the order they were added.
        objective: Coefficients of the objective, or ``None`` if it was never set.
        last_params: Parameters of the last solve, or ``None`` if it never ran.
        scripted: The result that :meth:`solve` returns. By default there is none, and the
            solve reports :attr:`BackendStatus.NO_SOLUTION`.
    """

    variables: list[RecordedVar] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    objective: Mapping[Var, float] | None = None
    last_params: SolveParams | None = None
    scripted: SolveResult | None = None

    def add_var(self, lb: float, ub: float, kind: VarKind, name: str = "") -> Var:
        """Record a variable and return its handle.

        Raises:
            ModelBuildError: If the bounds are reversed.
        """
        if lb > ub:
            raise ModelBuildError(
                f"variable {name!r}: lower bound {lb} above upper {ub}"
            )
        handle = len(self.variables)
        self.variables.append(RecordedVar(handle, lb, ub, kind, name))
        return handle

    def add_rows(self, rows: Iterable[Row]) -> None:
        """Record the rows, in order.

        Raises:
            ModelBuildError: If a row uses a variable that was never added: that is an error of
                the model, and the real solvers would report it only later, or not at all.
        """
        for row in rows:
            for var in row.terms:
                if not 0 <= var < len(self.variables):
                    raise ModelBuildError(
                        f"row {row.name!r} uses unknown variable {var}"
                    )
            self.rows.append(row)

    def set_objective(self, terms: Mapping[Var, float]) -> None:
        """Record the objective, replacing any previous one."""
        for var in terms:
            if not 0 <= var < len(self.variables):
                raise ModelBuildError(f"objective uses unknown variable {var}")
        self.objective = dict(terms)

    def solve(self, params: SolveParams) -> SolveResult:
        """Return the scripted result, and keep the parameters for inspection."""
        self.last_params = params
        return self.scripted or SolveResult(BackendStatus.NO_SOLUTION)

    def write(self, path: str) -> None:
        """Write the model as readable text, one row per line.

        It is not an LP file: it is meant for the eyes of whoever debugs a block.
        """
        lines = [
            f"x{v.handle} [{v.lb}, {v.ub}] {v.kind} {v.name}" for v in self.variables
        ]
        lines.append("minimise " + _format_terms(self.objective or {}))
        for row in self.rows:
            label = f"{row.name}: " if row.name else ""
            lines.append(f"{label}{_format_terms(row.terms)} {row.sense} {row.rhs:g}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_terms(terms: Mapping[Var, float]) -> str:
    if not terms:
        return "0"
    return " ".join(f"{c:+g}*x{v}" for v, c in terms.items())
