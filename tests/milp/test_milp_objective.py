"""Tests of the objective expressions: the coefficients are those of the scheduling weights."""

from __future__ import annotations

from typing import Any

import pytest
from sample import build, sample_instance

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.objective import response_epsilon
from satedge_workflow_scheduler.solvers.milp.blocks.base import ConstraintBlock
from satedge_workflow_scheduler.solvers.milp.blocks.delivery import (
    DeliveryBlock,
    DeliveryTimes,
)
from satedge_workflow_scheduler.solvers.milp.blocks.link_window import LinkWindowBlock
from satedge_workflow_scheduler.solvers.milp.blocks.path import PathBlock
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.tardiness import (
    TardinessBlock,
    TardinessTimes,
)
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData
from satedge_workflow_scheduler.solvers.milp.objective import (
    objective,
    rejection_cost,
    tardiness_cost,
)

ROUTED: list[ConstraintBlock[Any]] = [
    PathBlock(),
    StoreAndForwardBlock(),
    LinkWindowBlock(),
    DeliveryBlock(),
]


def test_rejection_cost_is_the_kappa_of_each_request() -> None:
    built = build(ROUTED)
    expr = rejection_cost(built.ctx)
    rejected = built.ctx.variables.rejected
    # κ(r1) = 2 (urgent, one non urgent request besides it), κ(r2) = 1
    assert expr.terms == {rejected["r1"]: 2.0, rejected["r2"]: 1.0}


def test_tardiness_cost_is_the_weight_of_each_request() -> None:
    built = build(
        [*ROUTED, TardinessBlock()],
        DeadlineMode.SOFT,
    )
    expr = tardiness_cost(built.ctx)
    tardy = built.model.families.get(TardinessTimes)
    # w(r1) = 1 + max(0, 600 − 550) = 51 (urgent), w(r2) = 1 (non urgent)
    assert expr.terms == {tardy.tard("r1"): 51.0, tardy.tard("r2"): 1.0}


def test_response_epsilon_is_the_smallest_cost_over_the_total_slack() -> None:
    # κ_min = 1, Σ(dl − rel) = (500 − 0) + (550 − 0)
    assert response_epsilon(sample_instance().requests) == pytest.approx(1.0 / 1051.0)
    assert ModelData(sample_instance(), DeadlineMode.HARD).epsilon == pytest.approx(
        1.0 / 1051.0
    )


def test_the_objective_is_the_rejection_cost_plus_epsilon_times_the_delivery() -> None:
    built = build(ROUTED)
    expr = objective(built.ctx)
    deliveries = built.model.families.get(DeliveryTimes)
    epsilon = 1.0 / 1051.0
    # release 0 in the sample, so the term is exactly ε δ_r, plus the rejection cost
    assert expr.terms[deliveries.deliv("r1")] == pytest.approx(epsilon)
    assert expr.terms[built.ctx.variables.rejected["r1"]] == pytest.approx(2.0)
