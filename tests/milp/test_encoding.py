"""Tests of the encoding: a calendar becomes values that the model accepts, and decodes back.

The calendar is the one of the priority-EDF/EFT heuristic on the sample instance, an independent
solver: the values must satisfy every row of the level-4 model whose variables are all encoded,
and decoding them must give the same calendar.
"""

from __future__ import annotations

from sample import Built, build, sample_instance

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.ports import BackendStatus
from satedge_workflow_scheduler.solvers.milp.config import Level, MilpConfig
from satedge_workflow_scheduler.solvers.milp.decoding import decode
from satedge_workflow_scheduler.solvers.milp.encoding import encode
from satedge_workflow_scheduler.solvers.milp.expr import Sense
from satedge_workflow_scheduler.solvers.milp.formulation_factory import blocks_for
from satedge_workflow_scheduler.solvers.priority_edf_eft.solver import (
    PriorityEdfEftSolver,
)

TOLERANCE = 1e-6


def _routed_model() -> Built:
    config = MilpConfig(level=Level.ROUTING, deadline=DeadlineMode.HARD)
    return build(blocks_for(config))


def test_the_heuristic_calendar_satisfies_the_routed_model() -> None:
    instance = sample_instance()
    solution = PriorityEdfEftSolver().solve(instance)
    built = _routed_model()
    values = encode(built.ctx.data, built.model, solution.schedule)

    checked = 0
    for row in built.backend.rows:
        if any(v not in values for v in row.terms):
            continue  # involves an auxiliary variable, which the calendar does not set
        lhs = sum(c * values[v] for v, c in row.terms.items())
        if row.sense is Sense.LE:
            assert lhs <= row.rhs + TOLERANCE, row.name
        elif row.sense is Sense.GE:
            assert lhs >= row.rhs - TOLERANCE, row.name
        else:
            assert abs(lhs - row.rhs) <= TOLERANCE, row.name
        checked += 1
    assert checked > 0


def test_the_encoded_calendar_decodes_back_to_the_same_calendar() -> None:
    instance = sample_instance()
    solution = PriorityEdfEftSolver().solve(instance)
    built = _routed_model()
    values = encode(built.ctx.data, built.model, solution.schedule)

    decoded = decode(
        instance,
        built.ctx.data,
        built.model,
        values,
        BackendStatus.OPTIMAL,
    )
    assert set(decoded.schedule.executions) == set(solution.schedule.executions)
    assert set(decoded.schedule.transfers) == set(solution.schedule.transfers)
