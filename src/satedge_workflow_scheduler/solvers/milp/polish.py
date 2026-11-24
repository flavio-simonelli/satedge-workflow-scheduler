"""The polish: re-solve the model with its binaries fixed, to remove the residuals of the big-M.

The integrality tolerance of a MIP solver lets a binary be, for example, 0.99999: the big-M terms
multiplied by that residual leave a small violation of the constraints. Fixing every binary to its
rounded value and solving again, as a linear program, gives the same schedule with exact values.

The polish is not part of the document: it is an implementation refinement, and it must not change
the optimum. So it is accepted only when the re-solve is optimal and its objective is no worse than
the one of the first solve, up to a tolerance; otherwise the first solution is kept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from satedge_workflow_scheduler.solvers.milp.backends.ports import (
    BackendStatus,
    MilpBackend,
    SolveParams,
    SolveResult,
)
from satedge_workflow_scheduler.solvers.milp.expr import Var

_RELATIVE_TOLERANCE = 1e-6
"""Relative tolerance on the objective: a polished solution may be this much worse, rounding aside."""


def _total_runtime(first: SolveResult, second: SolveResult) -> float | None:
    """The time of both solves, or ``None`` if the backend reports no time."""
    if first.runtime is None or second.runtime is None:
        return None
    return first.runtime + second.runtime


def polish(
    backend: MilpBackend,
    binaries: Sequence[Var],
    first: SolveResult,
    params: SolveParams,
) -> SolveResult:
    """The polished solution of ``first``, or ``first`` itself if the polish does not improve it.

    When the second solve runs, the runtime of the result is the time of both solves.

    Args:
        backend: The backend that solved the model; its bounds are changed by this function.
        binaries: The binary variables of the model.
        first: The result of the first solve, whose values are rounded and fixed.
        params: The parameters of the second solve.
    """
    if first.status not in (BackendStatus.OPTIMAL, BackendStatus.FEASIBLE):
        return first
    if first.objective is None:
        return first
    for var in binaries:
        value = round(first.values.get(var, 0.0))
        backend.set_bounds(var, float(value), float(value))
    second = backend.solve(params)
    runtime = _total_runtime(first, second)
    if second.status is not BackendStatus.OPTIMAL or second.objective is None:
        return replace(first, runtime=runtime)
    slack = _RELATIVE_TOLERANCE * (1.0 + abs(first.objective))
    if second.objective > first.objective + slack:
        return replace(first, runtime=runtime)
    return SolveResult(
        first.status,
        second.values,
        second.objective,
        second.gap,
        bound=first.bound,
        nodes=first.nodes,
        runtime=runtime,
    )
