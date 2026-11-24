"""Outcome of a solve together with the statistics of the solver."""

from __future__ import annotations

from dataclasses import dataclass

from satedge_workflow_scheduler.domain.solution import Solution


@dataclass(frozen=True)
class SolveStatistics:
    """What a solve cost, and how far its solution is from proven optimal.

    Attributes:
        method: Name of the method: ``"milp"`` or ``"sequential"``.
        wall_time_s: Seconds of wall time of the whole solve, or ``None`` if not measured.
        solver_time_s: Seconds the solver spent solving, summed over the rounds; ``None`` for a
            method that has no separate solver, such as the sequential heuristic.
        bound: Best bound on the objective found by the solver, for a MIP.
        gap: Relative gap between the solution and the bound, for a MIP.
        nodes: Branch-and-bound nodes explored, summed over the rounds; ``None`` if not a MIP.
        rounds: Number of models solved: one, unless the solve is lazy.
        binary_variables: Binary variables of the last model solved.
        constraints: Rows of the last model solved.
    """

    method: str
    wall_time_s: float | None = None
    solver_time_s: float | None = None
    bound: float | None = None
    gap: float | None = None
    nodes: float | None = None
    rounds: int = 1
    binary_variables: int | None = None
    constraints: int | None = None


@dataclass(frozen=True)
class SolverRun:
    """A solution together with the statistics of the solve that found it.

    Attributes:
        solution: The solution of the instance.
        statistics: The statistics of the solve.
    """

    solution: Solution
    statistics: SolveStatistics
