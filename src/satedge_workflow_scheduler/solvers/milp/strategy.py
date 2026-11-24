"""How the model is solved: in one solve, or with some constraints added lazily (rule 10).

The two strategies build and solve the same model, with the blocks of the configuration. They
differ in where the lazy groups of constraints are written:

- :class:`DirectSolve` writes every constraint from the start and solves once;
- :class:`LazySolve` solves without the lazy groups, finds the keys that the solution violates
  for each of them (for the PLMM, the overloaded satellites), writes the groups on those keys,
  and solves again, until nothing is violated. Each model of the loop is a relaxation of the
  complete one, so the first optimal solution that violates nothing is optimal for the complete
  model too.

A lazy group is a :class:`LazyConstraint`: it names its group of blocks and finds its violations
on a solution. Adding one means writing that class and making its blocks accept a subset of
their keys; the loop does not change.

Each round builds a new model instead of adding rows to the previous one: the builder stays
simple, and the solver restarts its search in either case. The reductions of the configuration
(:mod:`.reductions`) depend on the data only, so a solve computes them once, for all its rounds.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.solution import Schedule, Solution
from satedge_workflow_scheduler.scheduling.feasibility import FEASIBILITY_TOLERANCE
from satedge_workflow_scheduler.scheduling.plmm_occupancy import (
    load_profile,
    overloads,
    plmm_intervals,
)
from satedge_workflow_scheduler.solvers.errors import InfeasibleModelError, SolverError
from satedge_workflow_scheduler.solvers.milp.backends.ports import (
    BackendStatus,
    MilpBackend,
    SolveParams,
    SolveResult,
)
from satedge_workflow_scheduler.solvers.milp.blocks.base import BuildContext
from satedge_workflow_scheduler.solvers.milp.builder import BuiltModel, ModelBuilder
from satedge_workflow_scheduler.solvers.milp.config import LazyGroup, MilpConfig
from satedge_workflow_scheduler.solvers.milp.decoding import decode
from satedge_workflow_scheduler.solvers.milp.encoding import encode
from satedge_workflow_scheduler.solvers.milp.formulation_factory import (
    Restricted,
    blocks_for,
)
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData
from satedge_workflow_scheduler.solvers.milp.objective import objective
from satedge_workflow_scheduler.solvers.milp.polish import polish
from satedge_workflow_scheduler.solvers.milp.reductions import (
    Reductions,
    reductions_for,
)
from satedge_workflow_scheduler.solvers.run import SolverRun, SolveStatistics
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig
from satedge_workflow_scheduler.solvers.sequential.solver import SequentialSolver

BackendFactory = Callable[[], MilpBackend]


class SolveStrategy(Protocol):
    """A way to solve the model of a configuration on an instance."""

    def solve(
        self,
        instance: ProblemInstance,
        data: ModelData,
        config: MilpConfig,
        backend: BackendFactory,
    ) -> SolverRun:
        """The solution of ``instance``.

        Raises:
            InfeasibleModelError: If the model has no solution.
            SolverError: If the solver fails or stops without a valid solution.
        """
        ...


def warm_start_of(
    instance: ProblemInstance, data: ModelData, config: MilpConfig
) -> Schedule | None:
    """The calendar the solve starts from, or ``None`` if it starts from nothing.

    The calendar is the one of the sequential heuristic (priority-EDF, EFT, safe intervals), when
    the configuration asks for it. The heuristic reserves the committed calendar and the plan
    release, so the warm start is also given to online instances. A heuristic that cannot deliver
    every request in soft mode gives none.
    """
    if not config.warm_start:
        return None
    try:
        return (
            SequentialSolver(SequentialConfig(deadline=config.deadline))
            .solve(instance)
            .schedule
        )
    except InfeasibleModelError:
        return None


@dataclass(frozen=True)
class _Attempt:
    """One solve of a model: its solution, the result of the backend and the model."""

    solution: Solution
    result: SolveResult
    model: BuiltModel


def _statistics(attempts: Sequence[_Attempt]) -> SolveStatistics:
    """The statistics of a solve that solved ``attempts`` models.

    The times and the nodes add up over the rounds. The bound, the gap and the size of the model
    are those of the last round, whose solution is the one returned.
    """
    last = attempts[-1]
    runtimes = [a.result.runtime for a in attempts if a.result.runtime is not None]
    nodes = [a.result.nodes for a in attempts if a.result.nodes is not None]
    return SolveStatistics(
        method="milp",
        solver_time_s=sum(runtimes) if runtimes else None,
        bound=last.result.bound,
        gap=last.result.gap,
        nodes=sum(nodes) if nodes else None,
        rounds=len(attempts),
        binary_variables=len(last.model.binaries),
        constraints=sum(last.model.rows.values()),
    )


def solve_once(
    instance: ProblemInstance,
    data: ModelData,
    reductions: Reductions,
    config: MilpConfig,
    backend: BackendFactory,
    restricted: Restricted | None,
    params: SolveParams,
    start: Schedule | None = None,
) -> _Attempt:
    """Build the model with ``reductions``, with the lazy groups of ``restricted`` on their keys
    only, give it the starting calendar ``start`` if there is one, solve it once, and decode it.

    Raises:
        InfeasibleModelError: If the solver proves that the model has no solution.
        SolverError: If the solver fails, or stops without a solution.
    """
    sink = backend()
    blocks = blocks_for(config, online=data.is_online, restricted=restricted)
    model = ModelBuilder(data, sink, reductions).build(blocks)
    context = BuildContext(data, model.variables, model.reductions, model.families)
    sink.set_objective(dict(objective(context).terms))
    if start is not None:
        sink.set_start(encode(data, model, start))

    result = sink.solve(params)
    if config.polish:
        result = polish(sink, model.binaries, result, params)
    if result.status is BackendStatus.INFEASIBLE:
        raise InfeasibleModelError("the model has no solution")
    if result.status in (BackendStatus.ERROR, BackendStatus.NO_SOLUTION):
        raise SolverError(f"the solver ended with status {result.status}")
    solution = decode(instance, data, model, result.values, result.status)
    return _Attempt(solution, result, model)


class DirectSolve:
    """The whole model, with the PLMM on every satellite, in one solve."""

    def solve(
        self,
        instance: ProblemInstance,
        data: ModelData,
        config: MilpConfig,
        backend: BackendFactory,
    ) -> SolverRun:
        """The solution of ``instance``.

        Raises:
            InfeasibleModelError: If the model has no solution, or if the reductions prove
                that some request can never be served under soft deadlines.
            SolverError: If the solver fails or stops without a valid solution.
        """
        reductions = reductions_for(data, config.reductions)
        start = warm_start_of(instance, data, config)
        attempt = solve_once(
            instance, data, reductions, config, backend, None, config.solve, start
        )
        return SolverRun(attempt.solution, _statistics([attempt]))


class LazyConstraint(Protocol):
    """A group of constraints that the lazy strategy adds only where a solution violates it."""

    @property
    def group(self) -> LazyGroup:
        """The group of blocks that writes the constraints."""
        ...

    def violations(
        self, instance: ProblemInstance, solution: Solution
    ) -> frozenset[Hashable]:
        """The keys of the group that ``solution`` violates: where the blocks must be written."""
        ...


class LazyPlmm:
    """The capacity of the PLMM, added on the satellites that a solution overloads."""

    @property
    def group(self) -> LazyGroup:
        return LazyGroup.PLMM

    def violations(
        self, instance: ProblemInstance, solution: Solution
    ) -> frozenset[Hashable]:
        return frozenset(overloaded_satellites(instance, solution))


class LazySolve:
    """The lazy groups added only where a solution violates them (rule 10).

    Args:
        constraints: The lazy groups, one per group of blocks.

    Attributes:
        rounds: The keys of each lazy group written in each round of the last solve, in order.
            In the first round they are all empty.
    """

    def __init__(self, constraints: Sequence[LazyConstraint]) -> None:
        self._constraints = tuple(constraints)
        self.rounds: tuple[Mapping[LazyGroup, frozenset[Any]], ...] = ()

    def solve(
        self,
        instance: ProblemInstance,
        data: ModelData,
        config: MilpConfig,
        backend: BackendFactory,
    ) -> SolverRun:
        """The solution of the complete model, found by the lazy loop.

        The time limit of the configuration is for the whole loop: each round gets what is
        left. A round that is infeasible proves the complete model infeasible, because it is a
        relaxation of it.

        Raises:
            InfeasibleModelError: If a round has no solution, or if the reductions prove that
                some request can never be served under soft deadlines.
            SolverError: If the solver fails, if the time ends before a solution that violates
                nothing, or if a group is still violated where its constraints are written.
        """
        started = time.monotonic()
        reductions = reductions_for(data, config.reductions)
        start = warm_start_of(instance, data, config)
        written: dict[LazyGroup, frozenset[Any]] = {
            c.group: frozenset() for c in self._constraints
        }
        rounds: list[Mapping[LazyGroup, frozenset[Any]]] = []
        attempts: list[_Attempt] = []
        while True:
            rounds.append(dict(written))
            self.rounds = tuple(rounds)
            params = _remaining(config.solve, started)
            attempt = solve_once(
                instance, data, reductions, config, backend, written, params, start
            )
            attempts.append(attempt)
            solution = attempt.solution
            violated = {
                c.group: c.violations(instance, solution) for c in self._constraints
            }
            if not any(violated.values()):
                return SolverRun(solution, _statistics(attempts))
            new = {group: keys - written[group] for group, keys in violated.items()}
            if not any(new.values()):
                still = {g.value: sorted(map(str, k)) for g, k in violated.items() if k}
                raise SolverError(
                    f"{still} are violated although the model writes their constraints: "
                    "the model and the checker disagree"
                )
            for group, keys in new.items():
                written[group] |= keys


def overloaded_satellites(
    instance: ProblemInstance, solution: Solution
) -> frozenset[NodeId]:
    """The satellites whose PLMM the calendar of ``solution`` fills beyond its capacity.

    The occupation is computed on the calendar, committed requests included, as the checker
    computes it, and with the same tolerance.
    """
    schedule = solution.schedule
    occupancies = plmm_intervals(instance, schedule.executions, schedule.transfers)
    return frozenset(
        satellite.id
        for satellite in instance.network.satellites
        if overloads(
            load_profile(occupancies.get(satellite.id, ()), FEASIBILITY_TOLERANCE),
            satellite.plmm_capacity,
            FEASIBILITY_TOLERANCE,
        )
    )


_LAZY_CONSTRAINTS: dict[LazyGroup, Callable[[], LazyConstraint]] = {
    LazyGroup.PLMM: LazyPlmm,
}
"""The lazy implementation of each group."""


def strategy_for(config: MilpConfig) -> SolveStrategy:
    """The strategy that the configuration asks for: lazy if it has lazy groups."""
    if not config.lazy:
        return DirectSolve()
    groups = sorted(config.lazy, key=lambda group: group.value)
    return LazySolve([_LAZY_CONSTRAINTS[group]() for group in groups])


def _remaining(params: SolveParams, started: float) -> SolveParams:
    """The parameters of the next round, with the time that is left of the whole limit.

    Raises:
        SolverError: If no time is left.
    """
    if params.time_limit is None:
        return params
    left = params.time_limit - (time.monotonic() - started)
    if left <= 0:
        raise SolverError(
            "the time limit ended before a solution that violates nothing"
        )
    return replace(params, time_limit=left)
