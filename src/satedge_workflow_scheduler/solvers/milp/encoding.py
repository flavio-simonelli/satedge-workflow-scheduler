"""Encoding: from a calendar of the domain to the values of the variables of the model.

The inverse of :mod:`~satedge_workflow_scheduler.solvers.milp.decoding`. A calendar, whether the
solution of another solver or the committed calendar of an instance, becomes a value for each
decision variable and each routing variable: a task runs in the window that contains it, a data
item crosses an arc in the window that contains its transmission and latency (17b)–(17c), and
the requests that are not in the calendar are rejected.

The auxiliary variables of the blocks (the order of the PLMM, its flows, the delivery) are not
set: a calendar does not determine them, and the solver completes them. The encoding is therefore
a partial assignment, which is what a warm start accepts.

Only the variables of the model get a value: those that the reductions keep. A feasible calendar
uses only those, since the reductions remove what no feasible solution uses; a calendar that needs
a removed variable is not a calendar of this model, and the encoding refuses it.
"""

from __future__ import annotations

from collections.abc import Mapping

from satedge_workflow_scheduler.common.intervals import WindowSet
from satedge_workflow_scheduler.domain.solution import Schedule
from satedge_workflow_scheduler.scheduling.feasibility import FEASIBILITY_TOLERANCE
from satedge_workflow_scheduler.solvers.milp.blocks.path import RoutingVariables
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.builder import BuiltModel
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import Var
from satedge_workflow_scheduler.solvers.milp.model_data import (
    DataKey,
    ModelData,
    TaskKey,
)


def encode(
    data: ModelData, model: BuiltModel, schedule: Schedule
) -> Mapping[Var, float]:
    """The values of the decision and routing variables that describe ``schedule``.

    Every variable of those families gets a value: the ones the calendar does not use are 0, and
    a rejected request has ``ρ = 1``. Under soft deadlines there is no ``ρ``, and every request
    of the calendar is served.

    Raises:
        ModelBuildError: If an execution or a transmission lies in no window of its resource, or
            in a window or on an arc that the reductions removed, so the calendar is not a
            calendar of this model.
    """
    variables = model.variables
    values: dict[Var, float] = {}

    scheduled = schedule.request_ids
    for request_id, rho in variables.rejected.items():
        values[rho] = 0.0 if request_id in scheduled else 1.0

    for i in data.tasks:
        values[variables.t(i)] = 0.0
        for s in data.satellites:
            for w in variables.windows(i, s):
                values[variables.u(i, s, w)] = 0.0
    for execution in schedule.executions:
        i = TaskKey(execution.request_id, execution.task_index)
        what = f"execution of {i} on {execution.satellite}"
        window = _window_containing(
            data.hpc_windows(execution.satellite),
            execution.start,
            execution.end,
            what,
        )
        if window not in variables.windows(i, execution.satellite):
            raise ModelBuildError(f"{what} is in a window the reductions removed")
        values[variables.u(i, execution.satellite, window)] = 1.0
        values[variables.t(i)] = execution.start

    if model.families.has(RoutingVariables):
        _encode_routing(data, model, schedule, values)
    return values


def _encode_routing(
    data: ModelData,
    model: BuiltModel,
    schedule: Schedule,
    values: dict[Var, float],
) -> None:
    routing = model.families.get(RoutingVariables)
    starts = model.families.get(TransmissionStarts)
    for var in routing.crossed.values():
        values[var] = 0.0
    for by_window in routing.windows.values():
        for var in by_window.values():
            values[var] = 0.0
    for var in starts.start.values():
        values[var] = 0.0
    for transfer in schedule.transfers:
        a = DataKey(transfer.request_id, transfer.data_index)
        e = data.arc_between(transfer.source, transfer.target)
        what = f"transmission of {a} on arc {e}"
        if (a, e) in routing.removed:
            raise ModelBuildError(f"{what} is on an arc the reductions removed")
        values[starts.xi(a, e)] = transfer.start
        windows = data.link_windows(e)
        if windows is None:
            values[routing.crossed[(a, e)]] = 1.0
            continue
        # the window must contain the transmission and the latency, as in (17b)–(17c)
        end = transfer.end + data.latency(e)
        window = _window_containing(windows, transfer.start, end, what)
        if window not in routing.link_windows(a, e):
            raise ModelBuildError(f"{what} is in a window the reductions removed")
        values[routing.nu(a, e, window)] = 1.0


def _window_containing(windows: WindowSet, start: float, end: float, what: str) -> int:
    """The position of the window that contains ``[start, end]``, with the tolerance of the
    checker.

    Raises:
        ModelBuildError: If no window contains it.
    """
    for position, window in enumerate(windows):
        if (
            window.start - FEASIBILITY_TOLERANCE <= start
            and end <= window.end + FEASIBILITY_TOLERANCE
        ):
            return position
    raise ModelBuildError(f"{what} lies in no window")
