"""The priority-EDF / EFT constructive heuristic, as a solver of the common interface."""

from __future__ import annotations

from collections import defaultdict

from satedge_workflow_scheduler.common.errors import InfeasibleModelError
from satedge_workflow_scheduler.common.intervals import TIME_TOLERANCE
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.solution import (
    ObjectiveValue,
    RequestOutcome,
    Schedule,
    Solution,
    SolveStatus,
    TaskExecution,
    Transfer,
)
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.objective import evaluate
from satedge_workflow_scheduler.scheduling.plmm_occupancy import PlmmOccupancyInterval
from satedge_workflow_scheduler.solvers.priority_edf_eft.ledger import ResourceLedger
from satedge_workflow_scheduler.solvers.priority_edf_eft.ordering import (
    PriorityEdfOrdering,
    RequestOrdering,
)
from satedge_workflow_scheduler.solvers.priority_edf_eft.pending_raw import (
    earliest_raw_departure,
)
from satedge_workflow_scheduler.solvers.priority_edf_eft.placement import EftPlacement
from satedge_workflow_scheduler.solvers.priority_edf_eft.planner import (
    PlanFailure,
    RequestPlan,
    RequestPlanner,
)
from satedge_workflow_scheduler.solvers.priority_edf_eft.policies import (
    DeadlinePolicy,
    HardDeadlinePolicy,
    SoftDeadlinePolicy,
)
from satedge_workflow_scheduler.solvers.priority_edf_eft.routing import (
    EarliestArrivalRouter,
)


class PriorityEdfEftSolver:
    """Constructive heuristic: priority and EDF order, earliest-finish placement.

    The solver keeps no state between calls: each :meth:`solve` builds a fresh ledger.

    Args:
        mode: Deadline mode: hard (requests may be rejected) or soft (all requests served).
        ordering: Order in which requests are planned. Defaults to priority and EDF.
        eps: Tolerance for comparisons between instants.
        max_retries: Maximum number of retry instants after a PLMM failure (soft mode).
    """

    def __init__(
        self,
        mode: DeadlineMode = DeadlineMode.HARD,
        *,
        ordering: RequestOrdering | None = None,
        eps: float = TIME_TOLERANCE,
        max_retries: int = 100,
    ) -> None:
        self._mode = mode
        self._ordering = ordering if ordering is not None else PriorityEdfOrdering()
        self._eps = eps
        self._max_retries = max_retries

    def solve(self, instance: ProblemInstance) -> Solution:
        """Return a schedule for ``instance``.

        Raises:
            InfeasibleModelError: In soft mode, if some request cannot be delivered by the
                horizon. Its ``reasons`` gives the reason for each such request.
        """
        policy = self._policy()
        ledger = ResourceLedger(instance.network)
        router = EarliestArrivalRouter(instance.network, ledger, self._eps)
        placement = EftPlacement(instance.network, ledger, router, self._eps)
        planner = RequestPlanner(instance, ledger, placement, router, self._eps)

        pending = self._pending_raw(instance, policy)
        plans: dict[str, RequestPlan] = {}
        failures: dict[str, str] = {}

        for request in self._ordering.order(instance.requests):
            extra = self._extra_plmm(pending, request.id)
            attempts: list[float | None] = [None]
            attempts += policy.retry_instants(request, ledger, instance)
            result: RequestPlan | PlanFailure | None = None
            for not_before in attempts:
                with ledger.transaction() as tx:
                    attempt = planner.plan(
                        request,
                        limit=policy.limit(request, instance),
                        not_before=not_before,
                        tx=tx,
                        extra_plmm=extra,
                    )
                    if isinstance(attempt, RequestPlan):
                        tx.commit()
                result = attempt
                if isinstance(attempt, RequestPlan) or not attempt.plmm:
                    break

            if isinstance(result, RequestPlan):
                plans[request.id] = result
                pending.pop(request.id, None)
            elif result is not None:
                policy.on_failure(request, result)
                failures[request.id] = result.reason

        return self._solution(instance, plans, failures, policy)

    def _policy(self) -> DeadlinePolicy:
        if self._mode is DeadlineMode.HARD:
            return HardDeadlinePolicy()
        return SoftDeadlinePolicy(max_retries=self._max_retries)

    def _pending_raw(
        self, instance: ProblemInstance, policy: DeadlinePolicy
    ) -> dict[str, tuple[NodeId, PlmmOccupancyInterval]]:
        """Raw data of the pending requests, booked until ``e_r`` (soft mode only)."""
        pending: dict[str, tuple[NodeId, PlmmOccupancyInterval]] = {}
        if not policy.reserves_pending_raw():
            return pending
        for request in instance.requests:
            until = earliest_raw_departure(instance, request, self._eps)
            if until > request.release:
                item = PlmmOccupancyInterval(
                    request.id, 0, request.raw_size, request.release, until
                )
                pending[request.id] = (request.origin, item)
        return pending

    @staticmethod
    def _extra_plmm(
        pending: dict[str, tuple[NodeId, PlmmOccupancyInterval]], request_id: str
    ) -> dict[NodeId, list[PlmmOccupancyInterval]]:
        """Pending raw occupancies of the requests other than ``request_id``, by origin."""
        grouped: dict[NodeId, list[PlmmOccupancyInterval]] = defaultdict(list)
        for other_id, (origin, item) in pending.items():
            if other_id != request_id:
                grouped[origin].append(item)
        return grouped

    def _solution(
        self,
        instance: ProblemInstance,
        plans: dict[str, RequestPlan],
        failures: dict[str, str],
        policy: DeadlinePolicy,
    ) -> Solution:
        outcomes: list[RequestOutcome] = []
        executions: list[TaskExecution] = []
        transfers: list[Transfer] = []
        for request in instance.requests:
            plan = plans.get(request.id)
            if plan is None:
                outcomes.append(
                    RequestOutcome(
                        request.id,
                        accepted=False,
                        rejection_reason=failures.get(request.id),
                    )
                )
                continue
            outcomes.append(
                RequestOutcome(
                    request.id, accepted=True, delivery_time=plan.delivery_time
                )
            )
            executions.extend(plan.executions)
            transfers.extend(plan.transfers)

        if isinstance(policy, SoftDeadlinePolicy) and policy.reasons:
            raise InfeasibleModelError(
                "heuristic: some requests cannot be delivered by the horizon",
                policy.reasons,
            )

        value: ObjectiveValue = evaluate(instance, outcomes, self._mode)
        return Solution(
            schedule=Schedule(tuple(executions), tuple(transfers)),
            outcomes=tuple(outcomes),
            status=SolveStatus.FEASIBLE,
            objective=value,
        )
