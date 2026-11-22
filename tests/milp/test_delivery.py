"""Tests of the delivery and tardiness blocks: the rows must be (28)-(29), term by term."""

from __future__ import annotations

import pytest
from sample import Built, build

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.blocks.delivery import (
    DeliveryBlock,
    DeliveryTimes,
)
from satedge_workflow_scheduler.solvers.milp.blocks.link_window import LinkWindowBlock
from satedge_workflow_scheduler.solvers.milp.blocks.path import (
    PathBlock,
    RoutingVariables,
)
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.blocks.tardiness import (
    TardinessBlock,
    TardinessTimes,
)
from satedge_workflow_scheduler.solvers.milp.expr import Sense
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey

GROUND_ARC = 2  # S2 → G


def _hard() -> Built:
    return build(
        [PathBlock(), StoreAndForwardBlock(), LinkWindowBlock(), DeliveryBlock()]
    )


def test_one_delivery_row_per_arc_into_the_ground_station() -> None:
    # 2 requests, one arc into G: 2 rows
    assert _hard().model.rows["delivery"] == 2


def test_delivery_is_after_the_arrival_of_the_final_output() -> None:
    built = _hard()
    starts = built.model.families.get(TransmissionStarts)
    routing = built.model.families.get(RoutingVariables)
    deliveries = built.model.families.get(DeliveryTimes)
    final = DataKey("r1", 2)
    row = built.row(f"delivery[r1,{GROUND_ARC}]")
    # δ − ξ − (D + λ) χ ≥ 0, with χ the sum of the windows ν on this arc
    time_e = built.ctx.data.transmission_time(final, GROUND_ARC) + (
        built.ctx.data.latency(GROUND_ARC)
    )
    assert row.sense is Sense.GE
    assert row.rhs == pytest.approx(0.0)
    assert row.terms[deliveries.deliv("r1")] == pytest.approx(1.0)
    assert row.terms[starts.xi(final, GROUND_ARC)] == pytest.approx(-1.0)
    assert row.terms[routing.nu(final, GROUND_ARC, 0)] == pytest.approx(-time_e)


def test_tardiness_is_the_delivery_past_the_due_date() -> None:
    built = build(
        [
            PathBlock(),
            StoreAndForwardBlock(),
            LinkWindowBlock(),
            DeliveryBlock(),
            TardinessBlock(),
        ],
        DeadlineMode.SOFT,
    )
    assert built.model.rows["tardiness"] == 2
    deliveries = built.model.families.get(DeliveryTimes)
    tardy = built.model.families.get(TardinessTimes)
    row = built.row("tardiness[r1]")
    # τ − δ ≥ −dl, with dl = 500 for r1
    assert row.sense is Sense.GE
    assert row.rhs == pytest.approx(-500.0)
    assert row.terms[tardy.tard("r1")] == pytest.approx(1.0)
    assert row.terms[deliveries.deliv("r1")] == pytest.approx(-1.0)


def test_tardiness_is_bounded_by_the_horizon_past_the_due_date() -> None:
    built = build(
        [
            PathBlock(),
            StoreAndForwardBlock(),
            LinkWindowBlock(),
            DeliveryBlock(),
            TardinessBlock(),
        ],
        DeadlineMode.SOFT,
    )
    tardy = built.model.families.get(TardinessTimes)
    # horizon 600 − dl 500 = 100 for r1
    assert built.backend.variables[tardy.tard("r1")].ub == pytest.approx(100.0)
