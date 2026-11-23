from __future__ import annotations

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.scheduling.plmm_occupancy import PlmmOccupancyInterval
from satedge_workflow_scheduler.solvers.sequential.reservations import Reservations
from satedge_workflow_scheduler.solvers.sequential.routing.ports import (
    Route,
    RouteQuery,
)
from satedge_workflow_scheduler.solvers.sequential.routing.safe_interval import (
    SafeIntervalRouter,
)

HORIZON = 600.0


def _routes(
    instance: ProblemInstance, reservations: Reservations, target: str
) -> tuple[Route, ...]:
    query = RouteQuery(source="S1", held_since=0.0, ready=0.0, size=2.0, limit=HORIZON)
    return (
        SafeIntervalRouter().routes_from(instance, reservations, query).get(target, ())
    )


def test_with_room_everywhere_the_data_goes_as_early_as_possible() -> None:
    instance = sample_instance()
    (route,) = _routes(instance, Reservations(), "S2")
    arc = instance.network.arcs[instance.network.arc_index[("S1", "S2")]]
    assert route.arrival == pytest.approx(arc.transmission_time(2.0) + arc.latency)


def test_the_data_waits_where_the_next_plmm_has_room() -> None:
    instance = sample_instance()
    # S2 is full with 9 Gb until t = 20, so 2 Gb can enter it only from t = 20
    blocking = PlmmOccupancyInterval("other", 0, 9.0, start=0.0, end=20.0)
    reservations = Reservations().with_occupancies([("S2", blocking)])
    (route,) = _routes(instance, reservations, "S2")
    assert route.held_since == pytest.approx(20.0)
    assert route.arrival > 20.0


def test_the_source_without_room_sends_nothing() -> None:
    instance = sample_instance()
    full = PlmmOccupancyInterval("other", 0, 9.0, start=0.0, end=20.0)
    reservations = Reservations().with_occupancies([("S1", full)])
    query = RouteQuery(source="S1", held_since=0.0, ready=0.0, size=2.0, limit=HORIZON)
    routes = SafeIntervalRouter().routes_from(instance, reservations, query)
    assert set(routes) == {"S1"}
    assert routes["S1"][0].hops == ()


def test_only_the_final_output_enters_the_ground_station() -> None:
    instance = sample_instance()
    assert "G" not in SafeIntervalRouter().routes_from(
        instance,
        Reservations(),
        RouteQuery(source="S1", held_since=0.0, ready=0.0, size=2.0, limit=HORIZON),
    )


def test_routes_arriving_after_the_limit_are_discarded() -> None:
    instance = sample_instance()
    query = RouteQuery(source="S1", held_since=0.0, ready=0.0, size=2.0, limit=0.5)
    assert set(SafeIntervalRouter().routes_from(instance, Reservations(), query)) == {
        "S1"
    }
