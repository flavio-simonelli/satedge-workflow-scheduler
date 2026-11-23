"""Ports of the MILP backends: the contract between the model and a mathematical solver.

The model writes variables and rows through :class:`ModelSink`, which is all the constraint
blocks see. The strategy that solves the model and reads its values sees :class:`MilpBackend`,
which adds the objective, the solve and the results. No module outside ``backends`` imports a
solver library: each adapter implements these protocols.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from satedge_workflow_scheduler.solvers.milp.expr import Row, Var


class VarKind(StrEnum):
    """Domain of a variable."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


class BackendStatus(StrEnum):
    """Outcome of a solve, as the backend reports it."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    NO_SOLUTION = "no_solution"
    ERROR = "error"


@dataclass(frozen=True)
class SolveParams:
    """Parameters of one solve.

    Attributes:
        time_limit: Seconds of wall time, or ``None`` for no limit.
        mip_gap: Relative optimality gap at which the solve may stop.
        threads: Number of threads, or ``None`` to let the solver choose.
        memory_limit_gb: Memory the solver may use, in GB, or ``None`` for no limit. When it
            is reached the solver stops and returns its best solution so far.
        verbose: Whether the solver writes its log.
    """

    time_limit: float | None = None
    mip_gap: float = 1e-6
    threads: int | None = None
    memory_limit_gb: float | None = None
    verbose: bool = False


@dataclass(frozen=True)
class SolveResult:
    """What a backend returns after a solve.

    Attributes:
        status: Outcome of the solve.
        values: Value of each variable in the incumbent, empty if there is none.
        objective: Objective value of the incumbent, if there is one.
        gap: Relative gap reached, if the solver reports it.
    """

    status: BackendStatus
    values: Mapping[Var, float] = field(default_factory=dict)
    objective: float | None = None
    gap: float | None = None


class ModelSink(Protocol):
    """Where the model puts its variables and rows: the only view the constraint blocks have."""

    def add_var(self, lb: float, ub: float, kind: VarKind, name: str = "") -> Var:
        """Add a variable with bounds ``[lb, ub]`` and return its handle."""
        ...

    def add_rows(self, rows: Iterable[Row]) -> None:
        """Add the rows, in order. Adding them in batches is the expected use."""
        ...


class MilpBackend(ModelSink, Protocol):
    """A mathematical programming solver, seen as a minimal model builder and solver."""

    def set_objective(self, terms: Mapping[Var, float]) -> None:
        """Set the linear expression to minimise, given by its coefficients."""
        ...

    def set_bounds(self, var: Var, lb: float, ub: float) -> None:
        """Change the bounds of a variable, for a later solve (the polish fixes binaries)."""
        ...

    def set_start(self, values: Mapping[Var, float]) -> None:
        """Give the solver starting values for some variables (a warm start). Variables not in
        ``values`` are left to the solver. A starting point is a hint: the solver may ignore it,
        and it need not be feasible."""
        ...

    def solve(self, params: SolveParams) -> SolveResult:
        """Solve the model and return the outcome. It never raises on infeasibility."""
        ...

    def write(self, path: str) -> None:
        """Write the model to ``path`` (``.lp`` or ``.mps``), for inspection."""
        ...
