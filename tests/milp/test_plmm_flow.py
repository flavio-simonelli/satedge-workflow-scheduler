"""Tests of the PLMM blocks, flow formulation: the rows must be (20a)-(27), term by term."""

from __future__ import annotations

import pytest
from sample import Built, build

from satedge_workflow_scheduler.solvers.milp.blocks.link_window import LinkWindowBlock
from satedge_workflow_scheduler.solvers.milp.blocks.path import (
    PathBlock,
    RoutingVariables,
)
from satedge_workflow_scheduler.solvers.milp.blocks.plmm_flow import (
    FlowVariables,
    PlmmFlowBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.plmm_occupation import (
    PlmmOccupation,
    PlmmOccupationBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
    TransmissionStarts,
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
            PlmmFlowBlock(),
        ]
    )


def test_rows_of_the_occupation_and_of_the_flow() -> None:
    built = _built()
    # occupation (20a)–(21b): 6 + 2 + 10 + 6 + 15 = 39 rows
    assert built.model.rows["plmm occupation"] == 6 + 2 + 10 + 6 + 15
    # flow (22)–(27): 20 + 40 + 40 + 10 + 10 + 2 = 122 rows
    assert built.model.rows["plmm flow"] == 20 + 40 + 40 + 10 + 10 + 2


def test_variables_of_the_occupations_orders_and_flows() -> None:
    built = _built()
    occupation = built.model.families.get(PlmmOccupation)
    flow = built.model.families.get(FlowVariables)
    # ζ⁺ and ζ⁻: 5 × 2 each; π and f: 20 ordered pairs × 2; fini and ffin: 5 × 2 each; finifin: 2
    assert len(occupation.start) == 10
    assert len(occupation.end) == 10
    assert len(flow.before) == 40
    assert len(flow.flow) == 40
    assert len(flow.from_source) == 10
    assert len(flow.to_sink) == 10
    assert len(flow.unused) == 2


def test_the_capacity_of_each_plmm_is_split_between_sources_and_unused() -> None:
    built = _built()
    row = built.row("capacity[S1]")
    # Σ_b fini_{b,S1} + finifin_{S1} = K_{S1} = 10 Gb
    assert row.sense is Sense.EQ
    assert row.rhs == pytest.approx(10.0)
    assert len(row.terms) == 5 + 1


def test_a_data_received_on_a_satellite_starts_its_occupation_after_the_arrival() -> (
    None
):
    built = _built()
    occupation = built.model.families.get(PlmmOccupation)
    starts = built.model.families.get(TransmissionStarts)
    routing = built.model.families.get(RoutingVariables)
    a = DataKey("r1", 0)
    row = built.row(f"start-received[{a},1,S1]")
    # ξ_{a,1} + λ_1 + M(1 − χ_{a,1}) − ζ⁺_{a,S1} ≥ 0, with λ_1 = 1 s on S2→S1
    big_m = built.ctx.data.plmm_big_m
    assert row.sense is Sense.GE
    assert row.rhs == pytest.approx(-(1.0 + big_m))
    assert row.terms[starts.xi(a, 1)] == pytest.approx(1.0)
    assert row.terms[occupation.start[(a, "S1")]] == pytest.approx(-1.0)
    # on S2→S1 the arc is always available, so χ is a variable
    assert row.terms[routing.crossed[(a, 1)]] == pytest.approx(-big_m)
