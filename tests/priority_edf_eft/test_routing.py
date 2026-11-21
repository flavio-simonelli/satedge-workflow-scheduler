from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.intervals import TimeWindow, WindowSet
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Availability,
    Network,
    Plane,
    WindowedAvailability,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, Satellite
from satedge_workflow_scheduler.solvers.priority_edf_eft.ledger import (
    ArcKey,
    ResourceLedger,
)
from satedge_workflow_scheduler.solvers.priority_edf_eft.routing import (
    EarliestArrivalRouter,
)

EPS = 1e-9


def _sat(sid: str) -> Satellite:
    return Satellite(
        id=sid,
        plmm_capacity=10.0,
        hpc_rate=1.0,
        hpc_windows=WindowSet((TimeWindow(0.0, 1000.0),)),
    )


def _arc(source: str, target: str, availability: Availability) -> Arc:
    # capacity 1 Gbps and latency 1 s: a 2 Gb item takes 2 s to transmit
    return Arc(
        source=source,
        target=target,
        capacity=1.0,
        latency=1.0,
        availability=availability,
    )


def _setup(*arcs: Arc) -> tuple[Network, ResourceLedger, EarliestArrivalRouter]:
    sats = [_sat(sid) for sid in ("S1", "S2", "S3")]
    network = Network(
        planes=tuple(
            Plane(id=i, satellites=(s,), com=s.id) for i, s in enumerate(sats)
        ),
        ground_station=GroundStation(id="G"),
        arcs=arcs,
    )
    ledger = ResourceLedger(network)
    return network, ledger, EarliestArrivalRouter(network, ledger, EPS)


def _arc_by_name(network: Network, source: str, target: str) -> Arc:
    return network.arcs[network.arc_index[(source, target)]]


def test_always_available_arc_arrives_after_transmission_and_latency() -> None:
    network, _, router = _setup(_arc("S1", "S2", AlwaysAvailable()))
    arc = _arc_by_name(network, "S1", "S2")

    assert router.departure(arc, ready=5.0, size=2.0) == pytest.approx(5.0)
    routes = router.routes_from("S1", 5.0, 2.0, limit=100.0, to_ground=False)
    assert routes["S2"].arrival == pytest.approx(5.0 + 2.0 + 1.0)


def test_windowed_arc_waits_until_the_window_opens() -> None:
    windows = WindowedAvailability(WindowSet((TimeWindow(10.0, 20.0),)))
    network, _, router = _setup(_arc("S1", "S2", windows))
    arc = _arc_by_name(network, "S1", "S2")

    assert router.departure(arc, ready=0.0, size=2.0) == pytest.approx(10.0)


def test_window_too_short_for_transmission_and_latency_is_skipped() -> None:
    # the first window is 2 s long, but 2 s transmission + 1 s latency need 3 s
    windows = WindowedAvailability(
        WindowSet((TimeWindow(10.0, 12.0), TimeWindow(20.0, 30.0)))
    )
    network, _, router = _setup(_arc("S1", "S2", windows))
    arc = _arc_by_name(network, "S1", "S2")

    assert router.departure(arc, ready=0.0, size=2.0) == pytest.approx(20.0)


def test_windowed_arc_with_no_usable_window_has_no_departure() -> None:
    windows = WindowedAvailability(WindowSet((TimeWindow(10.0, 11.0),)))
    network, _, router = _setup(_arc("S1", "S2", windows))
    arc = _arc_by_name(network, "S1", "S2")

    assert router.departure(arc, ready=0.0, size=2.0) is None


def test_busy_arc_in_the_ledger_delays_the_departure() -> None:
    network, ledger, router = _setup(_arc("S1", "S2", AlwaysAvailable()))
    arc = _arc_by_name(network, "S1", "S2")
    with ledger.transaction() as tx:
        tx.reserve(ArcKey("S1", "S2"), TimeWindow(0.0, 10.0))
        tx.commit()
    assert router.departure(arc, ready=0.0, size=2.0) == pytest.approx(10.0)


def test_two_hops_beat_a_direct_arc_with_a_late_window() -> None:
    late = WindowedAvailability(WindowSet((TimeWindow(50.0, 100.0),)))
    _, _, router = _setup(
        _arc("S1", "S2", late),
        _arc("S1", "S3", AlwaysAvailable()),
        _arc("S3", "S2", AlwaysAvailable()),
    )

    routes = router.routes_from("S1", 0.0, 2.0, limit=200.0, to_ground=False)

    # direct: window opens at 50, arrival 53; via S3: arrival 3 then 6
    assert routes["S2"].arrival == pytest.approx(6.0)
    assert [(h.source, h.target) for h in routes["S2"].hops] == [
        ("S1", "S3"),
        ("S3", "S2"),
    ]


def test_ground_is_reached_only_when_requested() -> None:
    _, _, router = _setup(
        _arc("S1", "S2", AlwaysAvailable()),
        _arc("S2", "G", AlwaysAvailable()),
    )

    without = router.routes_from("S1", 0.0, 2.0, limit=100.0, to_ground=False)
    with_ground = router.routes_from("S1", 0.0, 2.0, limit=100.0, to_ground=True)

    assert "G" not in without
    assert with_ground["G"].arrival == pytest.approx(3.0 + 2.0 + 1.0)


def test_limit_discards_late_arrivals() -> None:
    _, _, router = _setup(_arc("S1", "S2", AlwaysAvailable()))

    assert "S2" in router.routes_from("S1", 0.0, 2.0, limit=3.0, to_ground=False)
    assert "S2" not in router.routes_from("S1", 0.0, 2.0, limit=2.0, to_ground=False)


def test_source_is_included_with_an_empty_route() -> None:
    _, _, router = _setup(_arc("S1", "S2", AlwaysAvailable()))
    routes = router.routes_from("S1", 4.0, 2.0, limit=100.0, to_ground=False)
    assert routes["S1"].hops == ()
    assert routes["S1"].arrival == pytest.approx(4.0)
