"""Tests of the arrival block: the rows must be (16a)-(16b), term by term."""

from __future__ import annotations

import pytest
from sample import Built, build

from satedge_workflow_scheduler.solvers.milp.blocks.arrival import ArrivalBlock
from satedge_workflow_scheduler.solvers.milp.blocks.path import PathBlock
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.expr import Sense
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey


def _built() -> Built:
    return build([PathBlock(), StoreAndForwardBlock(), ArrivalBlock()])


def test_rows_per_equation() -> None:
    # (16a): 3 data with a task consumer × 2 arcs into the satellites = 6;
    # (16b): 2 final data × 1 arc into the ground station = 2
    assert _built().model.rows["arrival"] == 6 + 2


def test_final_output_reaches_the_ground_station_before_its_limit() -> None:
    built = _built()
    starts = built.model.families.get(TransmissionStarts)
    final = DataKey("r1", 2)
    row = built.row(f"deliver[{final},2]")
    # τ̂ + c(1 − χ) − ξ − c ≥ 0 with τ̂ = 500: ξ_{a,2} + c χ_{a,2} ≤ 500
    assert row.sense is Sense.GE
    assert row.rhs == pytest.approx(-500.0)
    assert row.terms[starts.xi(final, 2)] == pytest.approx(-1.0)
