"""Tests of the PLMM event formulation: the rows must be the events of version B, term by term."""

from __future__ import annotations

import pytest
from sample import Built, build

from satedge_workflow_scheduler.solvers.milp.blocks.link_window import LinkWindowBlock
from satedge_workflow_scheduler.solvers.milp.blocks.path import PathBlock
from satedge_workflow_scheduler.solvers.milp.blocks.plmm_events import (
    EventVariables,
    PlmmEventsBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.plmm_occupation import (
    PlmmOccupation,
    PlmmOccupationBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
)
from satedge_workflow_scheduler.solvers.milp.expr import Sense
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey


def _built() -> Built:
    return build(
        [
            PathBlock(),
            StoreAndForwardBlock(),
            LinkWindowBlock(),
            PlmmOccupationBlock(),
            PlmmEventsBlock(),
        ]
    )


def test_rows_of_the_event_formulation() -> None:
    # 5 data, 2 satellites, 5 events each.
    # event order 4 × 2 = 8; presence (5 + 1) per data and satellite = 60;
    # contiguity (4 + 4) per data and satellite = 80; start (5 + 5) per data and satellite = 100;
    # end 4 per data and satellite = 40; capacity 5 × 2 = 10
    assert _built().model.rows["plmm events"] == 8 + 60 + 80 + 100 + 40 + 10


def test_there_is_one_date_per_event_and_satellite_and_one_occupation_per_event() -> (
    None
):
    built = _built()
    events = built.model.families.get(EventVariables)
    assert len(events.date) == 5 * 2
    assert len(events.on) == 5 * 2 * 5
    for var in built.backend.variables:
        if var.name.startswith("o["):
            assert (var.lb, var.ub) == (0.0, 1.0)


def test_the_events_of_a_satellite_do_not_decrease() -> None:
    built = _built()
    events = built.model.families.get(EventVariables)
    row = built.row("event-order[S1,0]")
    # τ_{S1,1} − τ_{S1,0} ≥ 0
    assert row.sense is Sense.GE
    assert row.rhs == pytest.approx(0.0)
    assert row.terms[events.date[("S1", 1)]] == pytest.approx(1.0)
    assert row.terms[events.date[("S1", 0)]] == pytest.approx(-1.0)


def test_the_start_of_the_occupation_is_the_date_of_the_event_where_it_begins() -> None:
    built = _built()
    events = built.model.families.get(EventVariables)
    occupation = built.model.families.get(PlmmOccupation)
    a = DataKey("r1", 0)
    row = built.row(f"start-up[{a},S1,0]")
    # ζ⁺_{a,S1} + M(1 − o_{a,S1,0}) − τ_{S1,0} ≥ 0 (the first event has no predecessor)
    big_m = built.ctx.data.plmm_big_m
    assert row.sense is Sense.GE
    assert row.rhs == pytest.approx(-big_m)
    assert row.terms[occupation.start[(a, "S1")]] == pytest.approx(1.0)
    assert row.terms[events.date[("S1", 0)]] == pytest.approx(-1.0)
