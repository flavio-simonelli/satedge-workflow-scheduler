from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Network,
    Plane,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, Satellite
from satedge_workflow_scheduler.solvers.priority_edf_eft.ledger import (
    HpcKey,
    ResourceLedger,
)
from satedge_workflow_scheduler.solvers.priority_edf_eft.placement import EftPlacement
from satedge_workflow_scheduler.solvers.priority_edf_eft.routing import (
    EarliestArrivalRouter,
)

EPS = 1e-9


def _sat(sid: str, windows: tuple[TimeWindow, ...]) -> Satellite:
    return Satellite(
        id=sid, plmm_capacity=10.0, hpc_write_rate=1.0, hpc_windows=windows
    )


def _arc(source: str, target: str) -> Arc:
    # capacity 1 Gbps, latency 1 s
    return Arc(
        source=source,
        target=target,
        capacity=1.0,
        latency=1.0,
        availability=AlwaysAvailable(),
    )


def _setup(
    s1_windows: tuple[TimeWindow, ...],
    s2_windows: tuple[TimeWindow, ...] = (TimeWindow(0.0, 1000.0),),
) -> tuple[Network, ResourceLedger, EftPlacement]:
    s1, s2 = _sat("S1", s1_windows), _sat("S2", s2_windows)
    network = Network(
        planes=(
            Plane(id=1, satellites=(s1,), com="S1"),
            Plane(id=2, satellites=(s2,), com="S2"),
        ),
        ground_station=GroundStation(id="G"),
        arcs=(_arc("S1", "S2"), _arc("S2", "S1")),
    )
    ledger = ResourceLedger(network)
    router = EarliestArrivalRouter(network, ledger, EPS)
    return network, ledger, EftPlacement(network, ledger, router, EPS)


def test_earliest_slot_starts_at_ready_when_inside_a_window() -> None:
    network, _, placement = _setup((TimeWindow(0.0, 100.0),))
    assert placement.earliest_slot(network.satellite("S1"), 5.0, 10.0) == pytest.approx(
        5.0
    )


def test_earliest_slot_waits_for_the_window_to_open() -> None:
    network, _, placement = _setup((TimeWindow(50.0, 100.0),))
    assert placement.earliest_slot(network.satellite("S1"), 0.0, 10.0) == pytest.approx(
        50.0
    )


def test_earliest_slot_skips_a_window_too_short_for_the_task() -> None:
    network, _, placement = _setup((TimeWindow(10.0, 15.0), TimeWindow(30.0, 100.0)))
    assert placement.earliest_slot(network.satellite("S1"), 0.0, 10.0) == pytest.approx(
        30.0
    )


def test_earliest_slot_avoids_hpc_reservations_in_the_ledger() -> None:
    network, ledger, placement = _setup((TimeWindow(0.0, 100.0),))
    with ledger.transaction() as tx:
        tx.reserve(HpcKey("S1"), TimeWindow(0.0, 50.0))
        tx.commit()
    assert placement.earliest_slot(network.satellite("S1"), 0.0, 10.0) == pytest.approx(
        50.0
    )


def test_earliest_slot_is_none_when_nothing_fits() -> None:
    network, _, placement = _setup((TimeWindow(0.0, 5.0),))
    assert placement.earliest_slot(network.satellite("S1"), 0.0, 10.0) is None


def test_task_goes_where_it_ends_first_even_if_its_input_arrives_later() -> None:
    # S1 can run only in [50, 100): the task would end at 60.
    # S2 receives 2 Gb after 3 s and runs at once: it ends at 13.
    _, _, placement = _setup((TimeWindow(50.0, 100.0),))
    chosen = placement.place("S1", ready=0.0, size=2.0, duration=10.0, limit=200.0)
    assert chosen is not None
    assert chosen.satellite == "S2"
    assert chosen.end == pytest.approx(13.0)


def test_ties_go_to_the_route_with_fewer_hops() -> None:
    # size 0: S2 receives the input after the latency (1 s), and S1 can start at 1 s too.
    # Both end at 11; S1 is reached with no hop, so it wins.
    _, _, placement = _setup((TimeWindow(1.0, 1000.0),))
    chosen = placement.place("S1", ready=0.0, size=0.0, duration=10.0, limit=200.0)
    assert chosen is not None
    assert chosen.satellite == "S1"
    assert chosen.input_route.hops == ()


def test_placement_is_none_when_every_end_exceeds_the_limit() -> None:
    _, _, placement = _setup((TimeWindow(50.0, 100.0),))
    assert placement.place("S1", ready=0.0, size=2.0, duration=10.0, limit=12.0) is None
