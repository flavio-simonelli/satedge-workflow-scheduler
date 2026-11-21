"""Independent check of a solution against the constraints of the final model.

Every solver, whatever its method, returns a :class:`Solution`. This module checks it
against the instance, family by family, in the order and with the numbers of the final
model, and returns the violations it finds. An empty result means the schedule is feasible.
It is the final check of the heuristic, the check of a committed calendar and the oracle of
the tests of every solver.

The check reads the schedule as the model does:

- a task occupies its HPC for ``[t_i, t_i + L̄_{i,s}]``;
- a transmission occupies its arc for ``[ξ_{a,e}, ξ_{a,e} + D_{a,e}]``, and its last bit
  arrives at ``ξ_{a,e} + D_{a,e} + λ_e``;
- each data item follows its transfers, sorted by start, from the node of its producer to
  the node of its consumer, or to the ground station for the final output.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import Arc, WindowedAvailability
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.domain.solution import (
    RequestOutcome,
    Solution,
    TaskExecution,
    Transfer,
)
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode, delivery_limit
from satedge_workflow_scheduler.scheduling.objective import evaluate
from satedge_workflow_scheduler.scheduling.plmm_occupancy import (
    load_profile,
    overloads,
    plmm_intervals,
)

FEASIBILITY_TOLERANCE = 1e-6
"""Tolerance in seconds and in Gb: a solver may return values rounded to this order."""


class Constraint(StrEnum):
    """The families of constraints of the final model, with their numbers."""

    OUTCOMES = "outcomes: one per request, none rejected with soft deadlines"
    ASSIGNMENT = "(2) assignment: one execution per task, on a satellite"
    TASK_DURATION = "task duration: the task lasts L̄_{i,s}"
    PRECEDENCE = "(3) precedence"
    RELEASE = "(4) release of the raw data"
    COMPUTING_WINDOW = "(5)-(7) computing window"
    HPC_OVERLAP = "(8)-(10) one task at a time on an HPC"
    PATH = "(11)-(13) data path"
    TRANSMISSION_TIME = "transmission time: the transfer lasts D_{a,e}"
    AFTER_PRODUCER = "(14a) transmission after the producer"
    AFTER_RELEASE = "(14b) raw data sent after its release"
    STORE_AND_FORWARD = "(15) store-and-forward"
    ARRIVAL = "(16a) consumer after the arrival"
    LIMIT = "(16b) delivery by the time limit"
    DELIVERY_TIME = "(28) delivery time"
    VISIBILITY_WINDOW = "(17a)-(17c) visibility window"
    LINK_CONTENTION = "(18)-(19) one transmission at a time on an arc"
    PLMM_CAPACITY = "(20)-(27) PLMM capacity"
    COMMITTED = "online: committed calendar unchanged"
    OBJECTIVE = "objective: the reported value is the one of the calendar"
    PLAN_RELEASE = "(30)-(31) plan release"


_STRUCTURAL = frozenset({Constraint.OUTCOMES, Constraint.ASSIGNMENT, Constraint.PATH})
"""Violations after which the PLMM occupancies cannot be built."""


@dataclass(frozen=True)
class Violation:
    """A constraint of the model that a solution does not respect.

    Attributes:
        constraint: The family of the constraint.
        request_id: The request concerned, if the violation concerns one.
        message: What is wrong, with the values involved.
    """

    constraint: Constraint
    request_id: str | None
    message: str

    def __str__(self) -> str:
        who = f"{self.request_id}: " if self.request_id is not None else ""
        return f"{self.constraint}: {who}{self.message}"


def check(
    instance: ProblemInstance,
    solution: Solution,
    mode: DeadlineMode,
    eps: float = FEASIBILITY_TOLERANCE,
) -> tuple[Violation, ...]:
    """Return the violations of the constraints of the model in ``solution``.

    Args:
        instance: The instance the solution is for.
        solution: The solution to check.
        mode: The deadline mode: it sets the time limit of each request and whether a
            request may be rejected.
        eps: Tolerance for comparisons between instants and between sizes.
    """
    schedule = _Schedule.of(instance, solution, mode, eps)
    families = (_outcomes, _structure, _tasks, _hpc_overlap, _data, _link_contention)
    violations = [v for family in families for v in family(schedule)]
    # the PLMM occupancies follow the data paths: they exist only if the paths do
    if not any(v.constraint in _STRUCTURAL for v in violations):
        violations += _plmm_capacity(schedule)
    violations += _online(schedule)
    if not any(v.constraint is Constraint.OUTCOMES for v in violations):
        violations += _objective(schedule)
    return tuple(violations)


@dataclass(frozen=True)
class _Schedule:
    """The solution read as the model reads it, with the indexes the checks need."""

    instance: ProblemInstance
    solution: Solution
    mode: DeadlineMode
    eps: float
    outcomes: dict[str, RequestOutcome]
    executions: dict[str, dict[int, TaskExecution]]
    hops: dict[tuple[str, int], list[Transfer]]

    @staticmethod
    def of(
        instance: ProblemInstance, solution: Solution, mode: DeadlineMode, eps: float
    ) -> _Schedule:
        executions: dict[str, dict[int, TaskExecution]] = defaultdict(dict)
        for e in solution.schedule.executions:
            executions[e.request_id][e.task_index] = e
        hops: dict[tuple[str, int], list[Transfer]] = defaultdict(list)
        for t in solution.schedule.transfers:
            hops[(t.request_id, t.data_index)].append(t)
        for chain in hops.values():
            chain.sort(key=lambda t: t.start)
        return _Schedule(
            instance,
            solution,
            mode,
            eps,
            {o.request_id: o for o in solution.outcomes},
            executions,
            hops,
        )

    @property
    def accepted(self) -> Iterator[Request]:
        """The accepted requests of the instance, in instance order."""
        for request in self.instance.requests:
            outcome = self.outcomes.get(request.id)
            if outcome is not None and outcome.accepted:
                yield request

    def arc(self, transfer: Transfer) -> Arc | None:
        index = self.instance.network.arc_index.get((transfer.source, transfer.target))
        return None if index is None else self.instance.network.arcs[index]


# --- outcomes ---------------------------------------------------------------------------


def _outcomes(s: _Schedule) -> Iterator[Violation]:
    ids = {r.id for r in s.instance.requests}
    for request_id in s.outcomes.keys() - ids:
        yield Violation(
            Constraint.OUTCOMES, request_id, "not a request of the instance"
        )
    for request in s.instance.requests:
        outcome = s.outcomes.get(request.id)
        if outcome is None:
            yield Violation(Constraint.OUTCOMES, request.id, "has no outcome")
        elif not outcome.accepted and s.mode is DeadlineMode.SOFT:
            yield Violation(
                Constraint.OUTCOMES, request.id, "rejected with soft deadlines"
            )


# --- structure: every item belongs to the pipeline of its request -----------------------


def _structure(s: _Schedule) -> Iterator[Violation]:
    requests = {r.id: r for r in s.instance.requests}
    for e in s.solution.schedule.executions:
        request = requests.get(e.request_id)
        if request is not None and not 0 <= e.task_index < len(request.tasks):
            yield Violation(
                Constraint.ASSIGNMENT,
                e.request_id,
                f"executes task {e.task_index}, but the pipeline has "
                f"{len(request.tasks)} tasks",
            )
    for t in s.solution.schedule.transfers:
        request = requests.get(t.request_id)
        if request is not None and not 0 <= t.data_index <= len(request.tasks):
            yield Violation(
                Constraint.PATH,
                t.request_id,
                f"sends data {t.data_index}, but the pipeline has "
                f"{len(request.tasks) + 1} data items",
            )
        elif s.arc(t) is None:
            yield Violation(
                Constraint.PATH,
                t.request_id,
                f"data {t.data_index}: no arc {t.source}->{t.target}",
            )


# --- tasks: (2)-(7) ---------------------------------------------------------------------


def _tasks(s: _Schedule) -> Iterator[Violation]:
    network = s.instance.network
    for request in s.accepted:
        done = s.executions.get(request.id, {})
        missing = sorted(set(range(len(request.tasks))) - done.keys())
        if missing:
            yield Violation(
                Constraint.ASSIGNMENT, request.id, f"tasks {missing} are not executed"
            )
        for k, e in sorted(done.items()):
            if k >= len(request.tasks):
                continue  # reported by the structure check
            if not network.is_satellite(e.satellite):
                yield Violation(
                    Constraint.ASSIGNMENT,
                    request.id,
                    f"task {k} runs on {e.satellite!r}, which is not a satellite",
                )
                continue
            satellite = network.satellite(e.satellite)
            duration = request.hpc_duration(k, satellite)
            if not math.isclose(e.end - e.start, duration, abs_tol=s.eps):
                yield Violation(
                    Constraint.TASK_DURATION,
                    request.id,
                    f"task {k} lasts {e.end - e.start} s on {e.satellite}, "
                    f"expected {duration} s",
                )
            span = TimeWindow(e.start, e.end)
            if not any(_contains(w, span, s.eps) for w in satellite.hpc_windows):
                yield Violation(
                    Constraint.COMPUTING_WINDOW,
                    request.id,
                    f"task {k} in [{e.start}, {e.end}] is in no computing window "
                    f"of {e.satellite}",
                )
        if 0 in done and done[0].start < request.release - s.eps:
            yield Violation(
                Constraint.RELEASE,
                request.id,
                f"task 0 starts at {done[0].start}, before the release "
                f"{request.release}",
            )
        for k in range(len(request.tasks) - 1):
            if k in done and k + 1 in done and done[k + 1].start < done[k].end - s.eps:
                yield Violation(
                    Constraint.PRECEDENCE,
                    request.id,
                    f"task {k + 1} starts at {done[k + 1].start}, before task {k} "
                    f"ends at {done[k].end}",
                )


# --- (8)-(10) ---------------------------------------------------------------------------


def _hpc_overlap(s: _Schedule) -> Iterator[Violation]:
    by_satellite: dict[NodeId, list[TaskExecution]] = defaultdict(list)
    for e in s.solution.schedule.executions:
        by_satellite[e.satellite].append(e)
    for satellite, items in by_satellite.items():
        items.sort(key=lambda e: e.start)
        for a, b in itertools.pairwise(items):
            if b.start < a.end - s.eps:
                yield Violation(
                    Constraint.HPC_OVERLAP,
                    b.request_id,
                    f"task {b.task_index} starts at {b.start} on {satellite}, while "
                    f"task {a.task_index} of {a.request_id} runs until {a.end}",
                )


# --- data: (11)-(17), (28) --------------------------------------------------------------


def _data(s: _Schedule) -> Iterator[Violation]:
    for request in s.accepted:
        done = s.executions.get(request.id, {})
        complete = set(done) == set(range(len(request.tasks)))
        on_satellites = all(
            s.instance.network.is_satellite(e.satellite) for e in done.values()
        )
        if not (complete and on_satellites):
            continue  # reported by (2): the data have no endpoints
        for k in range(len(request.tasks) + 1):
            yield from _data_item(s, request, k, done)


def _data_item(
    s: _Schedule, request: Request, k: int, done: dict[int, TaskExecution]
) -> Iterator[Violation]:
    """Path, timing and windows of the data item ``k`` of ``request``."""
    network = s.instance.network
    n = len(request.tasks)
    what = f"data {k}"
    source = request.origin if k == 0 else done[k - 1].satellite
    target = network.ground_station.id if k == n else done[k].satellite
    ready = request.release if k == 0 else done[k - 1].end
    chain = s.hops.get((request.id, k), [])

    node = source
    visited = {source}
    previous_arrival: float | None = None
    for hop in chain:
        arc = s.arc(hop)
        if arc is None:
            return  # reported by the structure check
        if hop.source != node:
            yield Violation(
                Constraint.PATH,
                request.id,
                f"{what}: leaves {hop.source} on {arc.name}, but it is on {node}",
            )
            return
        if hop.target in visited:
            yield Violation(
                Constraint.PATH, request.id, f"{what}: visits {hop.target} twice"
            )
            return
        if hop.target == network.ground_station.id and k < n:
            yield Violation(
                Constraint.PATH, request.id, f"{what}: only the final output goes down"
            )
        size = request.data_sizes[k]
        tx = arc.transmission_time(size)
        if not math.isclose(hop.end - hop.start, tx, abs_tol=s.eps):
            yield Violation(
                Constraint.TRANSMISSION_TIME,
                request.id,
                f"{what}: lasts {hop.end - hop.start} s on {arc.name}, expected {tx} s",
            )
        if previous_arrival is None:
            if hop.start < ready - s.eps:
                family = (
                    Constraint.AFTER_RELEASE if k == 0 else Constraint.AFTER_PRODUCER
                )
                yield Violation(
                    family,
                    request.id,
                    f"{what}: leaves {hop.source} at {hop.start}, before {ready}",
                )
        elif hop.start < previous_arrival - s.eps:
            yield Violation(
                Constraint.STORE_AND_FORWARD,
                request.id,
                f"{what}: leaves {hop.source} at {hop.start}, before it has arrived "
                f"at {previous_arrival}",
            )
        if isinstance(arc.availability, WindowedAvailability):
            span = TimeWindow(hop.start, hop.start + tx + arc.latency)
            if not any(_contains(w, span, s.eps) for w in arc.availability.windows):
                yield Violation(
                    Constraint.VISIBILITY_WINDOW,
                    request.id,
                    f"{what}: on {arc.name} in [{span.start}, {span.end}] is in no "
                    "visibility window",
                )
        previous_arrival = hop.start + tx + arc.latency
        node = hop.target
        visited.add(node)

    if node != target:
        yield Violation(
            Constraint.PATH, request.id, f"{what}: ends on {node} instead of {target}"
        )
        return
    arrival = ready if previous_arrival is None else previous_arrival
    if k < n:
        # with an empty path the consumer is on the producer's node: (3) or (4) apply
        if previous_arrival is not None and done[k].start < arrival - s.eps:
            yield Violation(
                Constraint.ARRIVAL,
                request.id,
                f"task {k} starts at {done[k].start}, before its input arrives at "
                f"{arrival}",
            )
        return
    yield from _delivery(s, request, arrival)


def _delivery(s: _Schedule, request: Request, arrival: float) -> Iterator[Violation]:
    limit = delivery_limit(request, s.mode, s.instance.horizon)
    if arrival > limit + s.eps:
        yield Violation(
            Constraint.LIMIT,
            request.id,
            f"the output reaches the ground at {arrival}, after the limit {limit}",
        )
    delivery = s.outcomes[request.id].delivery_time
    if delivery is None or not math.isclose(delivery, arrival, abs_tol=s.eps):
        yield Violation(
            Constraint.DELIVERY_TIME,
            request.id,
            f"the outcome says {delivery}, but the output reaches the ground at "
            f"{arrival}",
        )


# --- (18)-(19) --------------------------------------------------------------------------


def _link_contention(s: _Schedule) -> Iterator[Violation]:
    by_arc: dict[tuple[NodeId, NodeId], list[Transfer]] = defaultdict(list)
    for t in s.solution.schedule.transfers:
        by_arc[(t.source, t.target)].append(t)
    for (source, target), items in by_arc.items():
        items.sort(key=lambda t: t.start)
        for a, b in itertools.pairwise(items):
            if b.start < a.end - s.eps:
                yield Violation(
                    Constraint.LINK_CONTENTION,
                    b.request_id,
                    f"data {b.data_index} starts on {source}->{target} at {b.start}, "
                    f"while data {a.data_index} of {a.request_id} is sent until {a.end}",
                )


# --- (20)-(27) --------------------------------------------------------------------------


def _plmm_capacity(s: _Schedule) -> Iterator[Violation]:
    schedule = s.solution.schedule
    occupancies = plmm_intervals(s.instance, schedule.executions, schedule.transfers)
    for satellite in s.instance.network.satellites:
        profile = load_profile(occupancies.get(satellite.id, ()), s.eps)
        for segment in overloads(profile, satellite.plmm_capacity, s.eps):
            ids = sorted({item.request_id for item in segment.items})
            yield Violation(
                Constraint.PLMM_CAPACITY,
                None,
                f"PLMM of {satellite.id} holds {segment.load} Gb in "
                f"[{segment.start}, {segment.end}], over {satellite.plmm_capacity} Gb, "
                f"with data of {ids}",
            )


# --- online: committed calendar and (30)-(31) -------------------------------------------


def _online(s: _Schedule) -> Iterator[Violation]:
    instance = s.instance
    committed = instance.committed
    schedule = s.solution.schedule
    for request in instance.committed_requests:
        outcome = s.outcomes.get(request.id)
        if outcome is None or not outcome.accepted:
            yield Violation(Constraint.COMMITTED, request.id, "is not accepted")
            continue
        planned = schedule.restricted_to([request.id])
        fixed = committed.restricted_to([request.id])
        if set(planned.executions) != set(fixed.executions) or set(
            planned.transfers
        ) != set(fixed.transfers):
            yield Violation(
                Constraint.COMMITTED, request.id, "its calendar has been changed"
            )

    new = {r.id for r in instance.new_requests}
    for e in schedule.executions:
        if e.request_id in new:
            yield from _after_plan(
                s, e.request_id, f"task {e.task_index}", e.satellite, e.start
            )
    for t in schedule.transfers:
        if t.request_id in new:
            what = f"data {t.data_index} on {t.source}->{t.target}"
            yield from _after_plan(s, t.request_id, what, t.source, t.start)


def _after_plan(
    s: _Schedule, request_id: str, what: str, node: NodeId, start: float
) -> Iterator[Violation]:
    release = s.instance.release_of(node)
    if start < release - s.eps:
        reached = (
            "the plan never reaches it"
            if math.isinf(release)
            else (f"the plan reaches it at {release}")
        )
        yield Violation(
            Constraint.PLAN_RELEASE,
            request_id,
            f"{what} starts at {start} on {node}, but {reached}",
        )


# --- objective --------------------------------------------------------------------------


def _objective(s: _Schedule) -> Iterator[Violation]:
    """The objective reported by the solver, if any, is the one of its outcomes.

    The two terms are compared apart, with a relative tolerance: the weights of the
    tardiness can make the primary term large.
    """
    reported = s.solution.objective
    if reported is None:
        return
    expected = evaluate(s.instance, s.solution.outcomes, s.mode)
    terms = (
        ("primary", reported.primary, expected.primary),
        ("response", reported.response, expected.response),
    )
    for name, got, want in terms:
        if not math.isclose(got, want, rel_tol=s.eps, abs_tol=s.eps):
            yield Violation(
                Constraint.OBJECTIVE,
                None,
                f"the {name} term is reported as {got}, but the outcomes give {want}",
            )


def _contains(window: TimeWindow, span: TimeWindow, eps: float) -> bool:
    return window.start - eps <= span.start and span.end <= window.end + eps
