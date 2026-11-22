"""Tests of the link window block: the rows must be (17b)-(17c), term by term."""

from __future__ import annotations

import pytest
from sample import Built, build

from satedge_workflow_scheduler.solvers.milp.blocks.link_window import LinkWindowBlock
from satedge_workflow_scheduler.solvers.milp.blocks.path import (
    PathBlock,
    RoutingVariables,
)
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.expr import Sense
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey

WINDOWED_ARC = 2  # S2 → G, visible in [0, 1000]


def _built() -> Built:
    return build([PathBlock(), StoreAndForwardBlock(), LinkWindowBlock()])


def test_two_rows_per_transmission_on_a_windowed_arc() -> None:
    # 5 data, one windowed arc, (17b) and (17c): 10 rows; the others have no windows
    assert _built().model.rows["link window"] == 5 * 2


def test_the_transmission_starts_after_the_opening_of_its_window() -> None:
    built = _built()
    starts = built.model.families.get(TransmissionStarts)
    final = DataKey("r1", 2)
    row = built.row(f"open[{final},{WINDOWED_ARC}]")
    # 0 · ν − ξ ≤ 0: the window opens at 0, so the opening term is zero
    assert row.sense is Sense.LE
    assert row.rhs == pytest.approx(0.0)
    assert row.terms == {starts.xi(final, WINDOWED_ARC): pytest.approx(-1.0)}


def test_the_transmission_ends_before_the_closing_of_its_window() -> None:
    built = _built()
    starts = built.model.families.get(TransmissionStarts)
    routing = built.model.families.get(RoutingVariables)
    final = DataKey("r1", 2)
    row = built.row(f"close[{final},{WINDOWED_ARC}]")
    # ξ + (D + λ) Σ ν ≤ 1000 · ν, so the coefficient of ν is (D + λ) − 1000
    time_e = built.ctx.data.transmission_time(final, WINDOWED_ARC) + (
        built.ctx.data.latency(WINDOWED_ARC)
    )
    assert row.sense is Sense.LE
    assert row.rhs == pytest.approx(0.0)
    assert row.terms[starts.xi(final, WINDOWED_ARC)] == pytest.approx(1.0)
    assert row.terms[routing.nu(final, WINDOWED_ARC, 0)] == pytest.approx(
        time_e - 1000.0
    )
