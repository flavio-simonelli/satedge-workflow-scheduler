"""Tests of the store-and-forward block: the rows must be (14a)-(16b), term by term."""

from __future__ import annotations

import pytest
from sample import Built, build

from satedge_workflow_scheduler.solvers.milp.blocks.path import PathBlock
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.expr import Sense
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey


def _built() -> Built:
    return build([PathBlock(), StoreAndForwardBlock()])


def test_rows_per_equation() -> None:
    # (14a)+(14b): 5 data × 3 arcs = 15; (15): 5 data × (1 + 2) pairs at S1, S2 = 15;
    # (16a): 3 data with a consumer × 2 arcs into the satellites = 6;
    # (16b): 2 final data × 1 arc into the ground station = 2
    assert _built().model.rows["store and forward"] == 15 + 15 + 6 + 2


def test_raw_data_starts_at_its_release_on_every_arc() -> None:
    built = _built()
    starts = built.model.families.get(TransmissionStarts)
    raw = DataKey("r1", 0)
    row = built.row(f"raw[{raw},0]")
    # ξ_{a,0} − b_r χ_{a,0} ≥ 0, with b_r = 0 in the sample: the term b_r χ is dropped
    assert row.sense is Sense.GE
    assert row.rhs == pytest.approx(0.0)
    assert row.terms == {starts.xi(raw, 0): pytest.approx(1.0)}


def test_final_output_reaches_the_ground_station_before_its_limit() -> None:
    built = _built()
    starts = built.model.families.get(TransmissionStarts)
    final = DataKey("r1", 2)
    row = built.row(f"deliver[{final},2]")
    # τ̂ + c(1 − χ) − ξ − c ≥ 0 with τ̂ = 500: ξ_{a,2} + c χ_{a,2} ≤ 500
    assert row.sense is Sense.GE
    assert row.rhs == pytest.approx(-500.0)
    assert row.terms[starts.xi(final, 2)] == pytest.approx(-1.0)
