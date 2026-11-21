from __future__ import annotations

import itertools
from collections.abc import Callable

import pytest

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import WindowedAvailability
from satedge_workflow_scheduler.domain.solution import Solution
from satedge_workflow_scheduler.scheduling.plmm_occupancy import (
    load_profile,
    overloads,
    plmm_intervals,
)

EPS = 1e-6


def assert_valid(instance: ProblemInstance, solution: Solution) -> None:
    """Check the constraints of the model on a solution, failing with a message if violated."""
    network = instance.network
    requests = {r.id: r for r in instance.requests}
    executions = {(e.request_id, e.position): e for e in solution.schedule.executions}
    transfers = list(solution.schedule.transfers)

    for outcome in solution.outcomes:
        if not outcome.accepted:
            assert not any(
                e.request_id == outcome.request_id for e in solution.schedule.executions
            )
            continue
        request = requests[outcome.request_id]
        assert outcome.delivery_time is not None
        assert outcome.delivery_time <= instance.horizon + EPS

        # each task runs inside an HPC window, after its input has arrived
        previous_ready = request.release
        for position in range(len(request.tasks)):
            e = executions[(request.id, position)]
            satellite = network.satellite(e.satellite)
            assert any(
                w.start <= e.start + EPS and e.end <= w.end + EPS
                for w in satellite.hpc_windows
            ), f"task {request.id}/{position} outside HPC windows"
            assert e.start >= previous_ready - EPS, (
                "task starts before its input arrives"
            )
            assert e.end - e.start >= request.hpc_duration(position, satellite) - EPS
            previous_ready = e.end

    # no overlap on any HPC
    for sat in network.satellites:
        spans = sorted(
            (e.start, e.end)
            for e in solution.schedule.executions
            if e.satellite == sat.id
        )
        for (_, end_a), (start_b, _) in itertools.pairwise(spans):
            assert end_a <= start_b + EPS, f"HPC of {sat.id} runs two tasks at once"

    # no overlap on any arc, and each hop lies in a visibility window of its arc
    for arc in network.arcs:
        spans = sorted(
            (t.start, t.end)
            for t in transfers
            if (t.source, t.target) == (arc.source, arc.target)
        )
        for (_, end_a), (start_b, _) in itertools.pairwise(spans):
            assert end_a <= start_b + EPS, (
                f"arc {arc.name} carries two transfers at once"
            )
        if isinstance(arc.availability, WindowedAvailability):
            for start, end in spans:
                tx_end = end + arc.latency
                assert any(
                    w.start <= start + EPS and tx_end <= w.end + EPS
                    for w in arc.availability.windows
                ), f"transfer on {arc.name} outside its visibility windows"

    # PLMM capacity of every satellite, over the whole horizon
    occupancies = plmm_intervals(
        instance, solution.schedule.executions, solution.schedule.transfers
    )
    for sat in network.satellites:
        profile = load_profile(occupancies.get(sat.id, []))
        assert not overloads(profile, sat.plmm_capacity), f"PLMM of {sat.id} overflows"


@pytest.fixture
def check_valid() -> Callable[[ProblemInstance, Solution], None]:
    """Fixture giving tests access to :func:`assert_valid`."""
    return assert_valid
