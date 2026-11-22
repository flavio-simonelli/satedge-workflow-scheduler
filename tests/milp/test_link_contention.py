"""Tests of the link contention block: the rows must be (18)-(19), term by term."""

from __future__ import annotations

import pytest
from sample import Built, build

from satedge_workflow_scheduler.solvers.milp.blocks.link_contention import (
    LinkContentionBlock,
    LinkOrder,
)
from satedge_workflow_scheduler.solvers.milp.blocks.link_window import LinkWindowBlock
from satedge_workflow_scheduler.solvers.milp.blocks.path import PathBlock
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.expr import Sense
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey


def _built() -> Built:
    return build(
        [PathBlock(), StoreAndForwardBlock(), LinkWindowBlock(), LinkContentionBlock()]
    )


def test_two_rows_per_pair_of_data_and_arc() -> None:
    built = _built()
    # 6 pairs of data of different requests (3 × 2), 3 arcs, (18) and (19): 36 rows
    assert built.model.rows["link contention"] == 6 * 3 * 2
    assert len(built.model.families.get(LinkOrder).order) == 6 * 3


def test_the_order_of_the_pair_is_binary() -> None:
    built = _built()
    a, b = DataKey("r1", 0), DataKey("r2", 0)
    sigma = built.model.families.get(LinkOrder).sigma(a, b, 0)
    var = built.backend.variables[sigma]
    assert (var.lb, var.ub) == (0.0, 1.0)


def test_the_first_constraint_puts_b_after_a_when_a_precedes_b() -> None:
    built = _built()
    data = built.ctx.data
    starts = built.model.families.get(TransmissionStarts)
    order = built.model.families.get(LinkOrder)
    a, b = DataKey("r1", 0), DataKey("r2", 0)
    row = built.row(f"link-a[{a},{b},0]")
    tx_a = data.transmission_time(a, 0)
    tx_b = data.transmission_time(b, 0)
    big_m = max(data.limit("r1") + tx_a, data.limit("r2") + tx_b)
    # ξ_b − ξ_a − D_a + M(1 − σ) + M(2 − χ_a − χ_b) ≥ 0
    assert row.sense is Sense.GE
    assert row.terms[starts.xi(b, 0)] == pytest.approx(1.0)
    assert row.terms[starts.xi(a, 0)] == pytest.approx(-1.0)
    assert row.terms[order.sigma(a, b, 0)] == pytest.approx(-big_m)
    assert row.rhs == pytest.approx(tx_a - 3 * big_m)
