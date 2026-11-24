"""Adapter to Gurobi: the only module of the package that uses ``gurobipy``.

It translates the ports of :mod:`.ports` into calls to Gurobi, and Gurobi's outcomes into
:class:`BackendStatus`. The library is imported when the backend is created, so the package can
be imported without it: installing the ``gurobi`` extra is needed only to solve.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import Any

from satedge_workflow_scheduler.solvers.errors import SolverError
from satedge_workflow_scheduler.solvers.milp.backends.ports import (
    BackendStatus,
    SolveParams,
    SolveResult,
    VarKind,
)
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import (
    Row,
    Sense,
    Var,
    screen_empty_rows,
)


def _import_gurobipy() -> ModuleType:
    try:
        import gurobipy
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SolverError(
            "the Gurobi backend needs gurobipy: install the extra with `uv sync --extra gurobi`"
        ) from exc
    return gurobipy


class GurobiBackend:
    """Builds and solves the model with Gurobi.

    Variables are referred to by their position in creation order, which is their handle.
    Each run needs a fresh backend, because a Gurobi model holds one model only.
    """

    def __init__(self) -> None:
        self._gp = _import_gurobipy()
        self._model: Any = self._gp.Model()
        self._model.Params.OutputFlag = 0
        self._vars: list[Any] = []

    def add_var(self, lb: float, ub: float, kind: VarKind, name: str = "") -> Var:
        """Add a variable and return its handle.

        Raises:
            ModelBuildError: If a bound is NaN, which Gurobi would reject with its own error.
        """
        if math.isnan(lb) or math.isnan(ub):
            raise ModelBuildError(f"variable {name!r}: bounds must not be NaN")
        vtype = {
            VarKind.CONTINUOUS: self._gp.GRB.CONTINUOUS,
            VarKind.INTEGER: self._gp.GRB.INTEGER,
            VarKind.BINARY: self._gp.GRB.BINARY,
        }[kind]
        self._vars.append(self._model.addVar(lb=lb, ub=ub, vtype=vtype, name=name))
        return len(self._vars) - 1

    def add_rows(self, rows: Iterable[Row]) -> None:
        """Add the rows, in order.

        Raises:
            ModelBuildError: If a right-hand side or a coefficient is NaN.
        """
        grb = self._gp.GRB
        senses = {
            Sense.LE: grb.LESS_EQUAL,
            Sense.GE: grb.GREATER_EQUAL,
            Sense.EQ: grb.EQUAL,
        }
        for row in screen_empty_rows(rows):
            if math.isnan(row.rhs) or any(math.isnan(c) for c in row.terms.values()):
                raise ModelBuildError(
                    f"row {row.name!r} has a NaN coefficient or bound"
                )
            expr = self._linear(row.terms)
            self._model.addLConstr(expr, senses[row.sense], row.rhs, row.name)

    def set_objective(self, terms: Mapping[Var, float]) -> None:
        """Minimise the linear expression given by its coefficients."""
        self._model.setObjective(self._linear(terms), self._gp.GRB.MINIMIZE)

    def set_bounds(self, var: Var, lb: float, ub: float) -> None:
        """Change the bounds of ``var``, as the polish does before its second solve."""
        if math.isnan(lb) or math.isnan(ub):
            raise ModelBuildError("bounds must not be NaN")
        self._vars[var].LB = lb
        self._vars[var].UB = ub

    def set_start(self, values: Mapping[Var, float]) -> None:
        """Set the starting values of the variables, as the ``Start`` attribute of Gurobi.

        Raises:
            ModelBuildError: If a variable does not exist.
        """
        for var, value in values.items():
            if not 0 <= var < len(self._vars):
                raise ModelBuildError(f"starting value of unknown variable {var}")
            self._vars[var].Start = value

    def solve(self, params: SolveParams) -> SolveResult:
        """Solve the model and translate the outcome."""
        self._apply(params)
        self._model.optimize()
        return self._result()

    def write(self, path: str) -> None:
        """Write the model to ``path``; the format follows the extension (``.lp``, ``.mps``)."""
        self._model.write(path)

    def _linear(self, terms: Mapping[Var, float]) -> Any:
        for var in terms:
            if not 0 <= var < len(self._vars):
                raise ModelBuildError(f"expression uses unknown variable {var}")
        return self._gp.LinExpr(
            [float(c) for c in terms.values()],
            [self._vars[v] for v in terms],
        )

    def _apply(self, params: SolveParams) -> None:
        self._model.Params.OutputFlag = 1 if params.verbose else 0
        self._model.Params.MIPGap = params.mip_gap
        if params.time_limit is not None:
            self._model.Params.TimeLimit = params.time_limit
        if params.threads is not None:
            self._model.Params.Threads = params.threads
        if params.memory_limit_gb is not None:
            self._model.Params.SoftMemLimit = params.memory_limit_gb

    def _result(self) -> SolveResult:
        grb = self._gp.GRB.Status
        status = self._model.Status
        has_solution = self._model.SolCount > 0

        if status == grb.OPTIMAL:
            outcome = BackendStatus.OPTIMAL
        elif status == grb.INFEASIBLE:
            outcome = BackendStatus.INFEASIBLE
        elif status in (grb.INF_OR_UNBD, grb.UNBOUNDED):
            # unbounded or undecidable: an objective that can go to -infinity is a fault of
            # the model, since costs are always finite and the variables have finite bounds
            outcome = BackendStatus.ERROR
        elif has_solution:
            # stopped by a limit (time, memory, solutions) with an incumbent
            outcome = BackendStatus.FEASIBLE
        else:
            outcome = BackendStatus.NO_SOLUTION

        nodes = float(self._model.NodeCount) if self._model.IsMIP else None
        runtime = float(self._model.Runtime)
        if not has_solution:
            return SolveResult(outcome, nodes=nodes, runtime=runtime)

        values = {v: float(var.X) for v, var in enumerate(self._vars)}
        gap = float(self._model.MIPGap) if self._model.IsMIP else None
        bound = float(self._model.ObjBound) if self._model.IsMIP else None
        return SolveResult(
            outcome,
            values,
            objective=float(self._model.ObjVal),
            gap=gap,
            bound=bound,
            nodes=nodes,
            runtime=runtime,
        )
