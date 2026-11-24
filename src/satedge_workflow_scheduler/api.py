"""Public facade: solve instances from Python, from files, and plan the next epoch.

This is the interface other programs use, such as the constellation simulator. The command-line
interface is a thin layer over it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from satedge_workflow_scheduler.analysis import SolutionMetrics, analyze
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.domain.solution import Solution
from satedge_workflow_scheduler.io.files import load_instance, load_solver_config
from satedge_workflow_scheduler.io.results import write_result
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.gurobi import GurobiBackend
from satedge_workflow_scheduler.solvers.milp.config import Level, MilpConfig
from satedge_workflow_scheduler.solvers.milp.solver import MilpSolver
from satedge_workflow_scheduler.solvers.milp.strategy import BackendFactory
from satedge_workflow_scheduler.solvers.run import SolverRun, SolveStatistics
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig
from satedge_workflow_scheduler.solvers.sequential.solver import SequentialSolver

type SolverConfig = MilpConfig | SequentialConfig
"""The configuration of a solve: the MILP or the sequential heuristic."""


@dataclass(frozen=True)
class Result:
    """The outcome of one solve, with its statistics and its metrics.

    Attributes:
        instance: The instance that was solved.
        run: The solution and the statistics of the solve.
        metrics: The indicators of the solution, from :func:`analyze`.
        deadline: The deadline mode the solve used.
    """

    instance: ProblemInstance
    run: SolverRun
    metrics: SolutionMetrics
    deadline: DeadlineMode

    @property
    def solution(self) -> Solution:
        """The solution found."""
        return self.run.solution

    @property
    def statistics(self) -> SolveStatistics:
        """The statistics of the solve."""
        return self.run.statistics

    def save(self, path: str | Path) -> None:
        """Write the result as a JSON file at ``path``."""
        write_result(path, self.instance, self.run, self.deadline)


def solve(
    instance: ProblemInstance,
    config: SolverConfig | None = None,
    *,
    backend: BackendFactory | None = None,
) -> Result:
    """Solve ``instance`` with ``config``; by default the MILP at level 6.

    The MILP runs on Gurobi unless ``backend`` gives another backend; the sequential heuristic
    needs none.

    Raises:
        SolverError: If the solver fails, stops without a solution, or Gurobi is not installed.
        InfeasibleModelError: If the solver proves that there is no solution.
    """
    chosen = MilpConfig(level=Level.PLMM_CAPACITY) if config is None else config
    if isinstance(chosen, SequentialConfig):
        run = SequentialSolver(chosen).run(instance)
    else:
        run = MilpSolver(chosen, backend or GurobiBackend).run(instance)
    metrics = analyze(instance, run.solution, chosen.deadline)
    return Result(instance, run, metrics, chosen.deadline)


def solve_file(
    instance_path: str | Path,
    *,
    contact_plan: str | Path | None = None,
    online: str | Path | None = None,
    solver: str | Path | None = None,
    output: str | Path | None = None,
    backend: BackendFactory | None = None,
) -> Result:
    """Load an instance and solver configuration from files, solve, and optionally save.

    Raises:
        DocumentError: If one of the files cannot be read or is not valid.
        SolverError: As :func:`solve`.
    """
    instance = load_instance(instance_path, contact_plan=contact_plan, online=online)
    config = None if solver is None else load_solver_config(solver)
    result = solve(instance, config, backend=backend)
    if output is not None:
        result.save(output)
    return result


def compare(
    instance: ProblemInstance,
    configs: Mapping[str, SolverConfig],
    *,
    backend: BackendFactory | None = None,
) -> dict[str, Result]:
    """Solve ``instance`` once per configuration, keyed by the label of the configuration."""
    return {
        label: solve(instance, config, backend=backend)
        for label, config in configs.items()
    }


def next_epoch(
    instance: ProblemInstance,
    result: Result,
    *,
    now: float,
    new_requests: tuple[Request, ...] = (),
    plan_release: Mapping[str, float],
) -> ProblemInstance:
    """The instance of the next planning epoch, starting at ``now``.

    The accepted requests not yet delivered at ``now`` become committed: they keep their
    calendar from ``result``. The requests delivered before ``now`` and the rejected ones leave
    the instance. ``new_requests`` join it, and ``plan_release`` gives the plan release of the
    satellites in the next epoch.

    Raises:
        InstanceError: If the new requests repeat an identifier, or the plan release is invalid.
    """
    pending = frozenset(
        o.request_id
        for o in result.solution.outcomes
        if o.accepted and o.delivery_time is not None and o.delivery_time > now
    )
    kept = tuple(r for r in instance.requests if r.id in pending)
    return replace(
        instance,
        requests=(*kept, *new_requests),
        committed=result.solution.schedule.restricted_to(pending),
        plan_release=dict(plan_release),
    )


__all__ = [
    "Result",
    "SolverConfig",
    "compare",
    "next_epoch",
    "solve",
    "solve_file",
]
