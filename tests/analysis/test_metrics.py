"""Metrics of a solution: they agree with the schedule and the outcomes they are computed from."""

from __future__ import annotations

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.analysis import SolutionMetrics, analyze
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.requests import Priority
from satedge_workflow_scheduler.domain.solution import Solution
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig
from satedge_workflow_scheduler.solvers.sequential.solver import SequentialSolver


def _valid_metrics() -> tuple[ProblemInstance, Solution, SolutionMetrics]:
    instance = sample_instance()
    solution = SequentialSolver(SequentialConfig()).solve(instance)
    return instance, solution, analyze(instance, solution, DeadlineMode.HARD)


def test_one_entry_per_request_in_instance_order() -> None:
    instance, _, metrics = _valid_metrics()
    assert [r.request_id for r in metrics.requests] == [r.id for r in instance.requests]


def test_the_solution_of_the_sequential_heuristic_is_valid() -> None:
    _, _, metrics = _valid_metrics()
    assert metrics.violations == ()


def test_response_and_tardiness_follow_from_the_delivery() -> None:
    instance, solution, metrics = _valid_metrics()
    outcomes = {o.request_id: o for o in solution.outcomes}
    for entry in metrics.requests:
        outcome = outcomes[entry.request_id]
        assert entry.accepted == outcome.accepted
        if outcome.accepted:
            request = instance.request(entry.request_id)
            assert outcome.delivery_time is not None
            assert entry.response == pytest.approx(
                outcome.delivery_time - request.release
            )
            # in hard mode every accepted request is delivered by its limit
            assert entry.tardiness == pytest.approx(0.0)
        else:
            assert entry.response is None
            assert entry.tardiness is None


def test_counts_split_the_requests_by_priority() -> None:
    instance, _, metrics = _valid_metrics()
    for priority in Priority:
        total = sum(1 for r in instance.requests if r.priority is priority)
        accepted = metrics.count(priority, accepted=True)
        rejected = metrics.count(priority, accepted=False)
        assert accepted + rejected == total


def test_the_plmm_peak_is_within_capacity_on_a_valid_solution() -> None:
    instance, _, metrics = _valid_metrics()
    assert {p.satellite for p in metrics.plmm} == {
        s.id for s in instance.network.satellites
    }
    for entry in metrics.plmm:
        assert entry.peak <= entry.capacity + 1e-9


def test_hpc_busy_time_is_the_sum_of_the_executions() -> None:
    _, solution, metrics = _valid_metrics()
    for entry in metrics.hpc:
        expected = sum(
            e.end - e.start
            for e in solution.schedule.executions
            if e.satellite == entry.satellite
        )
        assert entry.busy == pytest.approx(expected)
        assert 0.0 <= entry.utilisation <= 1.0 + 1e-9


def test_arc_volume_is_the_size_of_the_transferred_data() -> None:
    instance, solution, metrics = _valid_metrics()
    expected = sum(
        instance.request(t.request_id).data_sizes[t.data_index]
        for t in solution.schedule.transfers
    )
    assert sum(a.volume for a in metrics.arcs) == pytest.approx(expected)
