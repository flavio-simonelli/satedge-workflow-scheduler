"""Decoding: from the values of a solve to a :class:`Solution` of the domain.

The decoder reads the values and nothing else: it does not know the constraints, and it does not
judge the solution. The calendar of the new requests comes from the routing and the timing
variables. A task runs on the satellite where its ``x`` is 1, from its start ``t``, and lasts its
duration there. A data item crosses an arc where its ``χ`` is 1, from its start ``ξ``, and lasts
the transmission time. The calendar of the committed requests is copied from the instance: the
model keeps it fixed, and copying it avoids the rounding of the solver.

The delivery of an accepted request is read from the calendar, as the checker does: the arrival of
its final output at the ground station. A rejection is read from ``ρ``.

The objective is not read from the solver: it is evaluated on the outcomes, as the scheduling
module does, so that the value reported is the value of the schedule. Checking the solution is
the job of :func:`satedge_workflow_scheduler.scheduling.feasibility.check`, not of the decoder.
"""

from __future__ import annotations

from collections.abc import Mapping

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.solution import (
    RequestOutcome,
    Schedule,
    Solution,
    SolveStatus,
    TaskExecution,
    Transfer,
)
from satedge_workflow_scheduler.scheduling.objective import evaluate
from satedge_workflow_scheduler.solvers.errors import SolverError
from satedge_workflow_scheduler.solvers.milp.backends.ports import BackendStatus
from satedge_workflow_scheduler.solvers.milp.blocks.delivery import DeliveryTimes
from satedge_workflow_scheduler.solvers.milp.blocks.path import RoutingVariables
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.builder import BuiltModel
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Var
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData

_BINARY_THRESHOLD = 0.5
"""A binary variable is 1 when its value is above one half: a solver returns 1e-9 for a zero."""


def decode(
    instance: ProblemInstance,
    data: ModelData,
    model: BuiltModel,
    values: Mapping[Var, float],
    status: BackendStatus,
) -> Solution:
    """The solution described by the values of a solve.

    Raises:
        ValueError: If ``status`` is not one that comes with a solution.
    """
    if status not in (BackendStatus.OPTIMAL, BackendStatus.FEASIBLE):
        raise ValueError(f"no solution to decode with status {status}")
    new = _schedule(data, model, values)
    committed = instance.committed
    schedule = Schedule(
        (*committed.executions, *new.executions), (*committed.transfers, *new.transfers)
    )
    outcomes = _outcomes(data, model, values, schedule)
    objective = evaluate(instance, outcomes, data.mode)
    solve_status = (
        SolveStatus.OPTIMAL if status is BackendStatus.OPTIMAL else SolveStatus.FEASIBLE
    )
    return Solution(schedule, outcomes, solve_status, objective)


def _value(expr: LinExpr, values: Mapping[Var, float]) -> float:
    return sum(c * values.get(v, 0.0) for v, c in expr.terms.items()) + expr.constant


def _schedule(
    data: ModelData, model: BuiltModel, values: Mapping[Var, float]
) -> Schedule:
    variables = model.variables
    routed = model.families.has(RoutingVariables)

    executions: list[TaskExecution] = []
    for i in data.tasks:
        if data.is_committed(i.request_id):
            continue
        for s in data.satellites:
            if _value(variables.x(i, s), values) > _BINARY_THRESHOLD:
                start = values.get(variables.t(i), 0.0)
                executions.append(
                    TaskExecution(
                        i.request_id,
                        i.task_index,
                        s,
                        start,
                        start + data.duration(i, s),
                    )
                )

    transfers: list[Transfer] = []
    if not routed:
        return Schedule(tuple(executions), ())
    starts = model.families.get(TransmissionStarts)
    routing = model.families.get(RoutingVariables)
    for a in data.data:
        if data.is_committed(a.request_id):
            continue
        for e in data.arc_ids:
            if _value(routing.chi(a, e), values) <= _BINARY_THRESHOLD:
                continue
            source, target = data.endpoints(e)
            start = values.get(starts.xi(a, e), 0.0)
            transfers.append(
                Transfer(
                    a.request_id,
                    a.data_index,
                    source,
                    target,
                    start,
                    start + data.transmission_time(a, e),
                )
            )
    return Schedule(tuple(executions), tuple(transfers))


def _outcomes(
    data: ModelData,
    model: BuiltModel,
    values: Mapping[Var, float],
    schedule: Schedule,
) -> tuple[RequestOutcome, ...]:
    outcomes: list[RequestOutcome] = []
    routed = model.families.has(RoutingVariables)
    for r in data.request_ids:
        rejected = not data.is_committed(r) and (
            _value(model.variables.rejection(r), values) > _BINARY_THRESHOLD
        )
        if rejected:
            outcomes.append(RequestOutcome(r, accepted=False))
        else:
            if routed:
                delivery = _arrival_at_ground(data, schedule, r)
            else:
                # without routing, the delivery is the variable δ of the fixed delays
                delivery = values.get(model.families.get(DeliveryTimes).deliv(r), 0.0)
            outcomes.append(RequestOutcome(r, accepted=True, delivery_time=delivery))
    return tuple(outcomes)


def _arrival_at_ground(data: ModelData, schedule: Schedule, request_id: str) -> float:
    """The instant the final output of the request reaches the ground station.

    Raises:
        SolverError: If no transmission brings the final output to the ground station.
    """
    final = data.final_data(request_id)
    arrivals = [
        t.start
        + data.transmission_time(final, data.arc_between(t.source, t.target))
        + data.latency(data.arc_between(t.source, t.target))
        for t in schedule.transfers
        if t.request_id == request_id
        and t.data_index == final.data_index
        and t.target == data.ground_station
    ]
    if not arrivals:
        raise SolverError(f"the solution delivers no final output of {request_id}")
    return max(arrivals)
