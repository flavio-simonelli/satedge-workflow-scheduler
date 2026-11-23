"""The sequential heuristic as a solver of the common interface.

The skeleton is the same for every member of the family:

1. the new requests are sorted (:mod:`.ordering`);
2. in soft mode, the raw data of every new request is reserved on its origin (:mod:`.pending_raw`);
3. each request, in that order, is screened (:func:`unrealizable_reason`) and then planned around
   the reservations of the requests before it (:mod:`.placement`), whose routes come from
   :mod:`.routing`;
4. a request that does not fit is rejected in hard mode, and makes the instance infeasible in
   soft mode, where nothing can be rejected;
5. the calendar is checked against the model before it is returned, if the configuration asks.

The choices of the family come from :class:`SequentialConfig`; the solver only assembles them.
"""

from __future__ import annotations

from collections.abc import Mapping

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.solution import (
    RequestOutcome,
    Schedule,
    Solution,
    SolveStatus,
)
from satedge_workflow_scheduler.scheduling.bounds import request_bounds
from satedge_workflow_scheduler.scheduling.deadline import (
    DeadlineMode,
    delivery_limit,
    may_reject,
)
from satedge_workflow_scheduler.scheduling.feasibility import check
from satedge_workflow_scheduler.scheduling.objective import evaluate
from satedge_workflow_scheduler.scheduling.screening import unrealizable_reason
from satedge_workflow_scheduler.solvers.errors import InfeasibleModelError, SolverError
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig
from satedge_workflow_scheduler.solvers.sequential.errors import SequentialConfigError
from satedge_workflow_scheduler.solvers.sequential.factory import (
    ordering_for,
    router_for,
    scheduler_for,
)
from satedge_workflow_scheduler.solvers.sequential.pending_raw import (
    pending_raw_occupancy,
)
from satedge_workflow_scheduler.solvers.sequential.placement.ports import (
    PlanFailure,
    RequestPlan,
)
from satedge_workflow_scheduler.solvers.sequential.reservations import Reservations


class SequentialSolver:
    """The sequential heuristic: order the requests, then plan them one at a time.

    The solver keeps no state between calls: each :meth:`solve` builds its own reservations.

    Args:
        config: The members of the family to use, and the deadline mode.

    Raises:
        SequentialConfigError: At solve time, for instances with committed requests or a plan
            release: they are not supported yet.
    """

    def __init__(self, config: SequentialConfig) -> None:
        self._config = config

    def solve(self, instance: ProblemInstance) -> Solution:
        """Return a schedule for ``instance``.

        Raises:
            InfeasibleModelError: In soft mode, if some request cannot be delivered by the
                horizon. Its ``reasons`` gives the reason for each such request.
            SolverError: If the calendar violates the model, which is a bug.
        """
        config = self._config
        _reject_online(instance)
        mode = config.deadline
        ordering = ordering_for(config.ordering)
        router = router_for(config.routing, config.eps)
        scheduler = scheduler_for(config, router)

        reservations = Reservations()
        if not may_reject(mode):
            for request in instance.new_requests:
                pending = pending_raw_occupancy(instance, request, config.eps)
                if pending is not None:
                    reservations = reservations.with_occupancies(
                        [(request.origin, pending)]
                    )

        plans: dict[str, RequestPlan] = {}
        failures: dict[str, str] = {}
        for request in ordering.order(instance.new_requests):
            limit = delivery_limit(request, mode, instance.horizon)
            bounds = request_bounds(instance, request, limit, config.eps)
            reason = unrealizable_reason(request, bounds, instance.network)
            if reason is not None:
                failures[request.id] = reason
                continue
            base = reservations.without_occupancies_of(request.id)
            result = scheduler.schedule(instance, request, bounds, base)
            if isinstance(result, PlanFailure):
                failures[request.id] = result.reason
                continue
            plans[request.id] = result
            reservations = base.with_schedule(instance, result.schedule)

        if failures and not may_reject(mode):
            raise InfeasibleModelError(
                "heuristic: some requests cannot be delivered by the horizon", failures
            )
        solution = _solution(instance, plans, failures, mode)
        if config.verify:
            violations = check(instance, solution, mode)
            if violations:
                lines = "\n".join(str(v) for v in violations)
                raise SolverError(
                    f"sequential heuristic: the calendar violates the model:\n{lines}"
                )
        return solution


def _reject_online(instance: ProblemInstance) -> None:
    """Refuse the instances with committed requests or a plan release: not supported yet."""
    if instance.committed.request_ids or any(
        release > 0 for release in instance.plan_release.values()
    ):
        raise SequentialConfigError(
            "the sequential heuristic does not handle committed requests or a plan release yet"
        )


def _solution(
    instance: ProblemInstance,
    plans: Mapping[str, RequestPlan],
    failures: Mapping[str, str],
    mode: DeadlineMode,
) -> Solution:
    """The outcomes in instance order, and the calendar of the planned requests."""
    outcomes: list[RequestOutcome] = []
    for request in instance.requests:
        if request.id in plans:
            outcomes.append(
                RequestOutcome(request.id, True, plans[request.id].delivery_time)
            )
        else:
            outcomes.append(
                RequestOutcome(
                    request.id, False, rejection_reason=failures.get(request.id)
                )
            )
    planned = [plans[r.id].schedule for r in instance.requests if r.id in plans]
    schedule = Schedule(
        tuple(e for s in planned for e in s.executions),
        tuple(t for s in planned for t in s.transfers),
    )
    return Solution(
        schedule=schedule,
        outcomes=tuple(outcomes),
        status=SolveStatus.FEASIBLE,
        objective=evaluate(instance, outcomes, mode),
    )


__all__ = ["SequentialSolver"]
