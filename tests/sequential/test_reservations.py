from __future__ import annotations

from satedge_workflow_scheduler.common.intervals import TimeWindow, WindowSet
from satedge_workflow_scheduler.domain.nodes import Satellite
from satedge_workflow_scheduler.domain.solution import TaskExecution, Transfer
from satedge_workflow_scheduler.scheduling.plmm_occupancy import PlmmOccupancyInterval
from satedge_workflow_scheduler.solvers.sequential.reservations import Reservations

EPS = 1e-9


def _sat(capacity: float = 10.0) -> Satellite:
    return Satellite(
        id="S1",
        plmm_capacity=capacity,
        hpc_rate=1.0,
        hpc_windows=WindowSet((TimeWindow(0.0, 100.0),)),
    )


def test_an_empty_plmm_has_room_for_a_stay_within_capacity() -> None:
    assert Reservations().plmm_has_room(_sat(), 4.0, 0.0, 50.0, EPS)


def test_a_stay_with_no_room_during_it_is_refused() -> None:
    item = PlmmOccupancyInterval("r1", 0, 8.0, start=10.0, end=20.0)
    reserved = Reservations().with_occupancies([("S1", item)])
    assert not reserved.plmm_has_room(_sat(), 4.0, 15.0, 18.0, EPS)
    assert reserved.plmm_has_room(_sat(), 4.0, 0.0, 10.0, EPS)  # ends when item starts
    assert reserved.plmm_has_room(_sat(), 4.0, 20.0, 30.0, EPS)  # starts when it ends


def test_reserving_does_not_change_the_receiver() -> None:
    base = Reservations()
    base.with_execution(TaskExecution("r1", 0, "S1", 0.0, 5.0))
    assert base.hpc_busy("S1").intervals == ()


def test_an_execution_busies_its_hpc() -> None:
    reserved = Reservations().with_execution(TaskExecution("r1", 0, "S1", 2.0, 5.0))
    assert reserved.hpc_busy("S1").intervals == (TimeWindow(2.0, 5.0),)
    assert reserved.hpc_busy("S2").intervals == ()


def test_a_transfer_busies_its_arc_only() -> None:
    reserved = Reservations().with_transfer(Transfer("r1", 0, "S1", "S2", 1.0, 3.0))
    assert reserved.arc_busy("S1", "S2").intervals == (TimeWindow(1.0, 3.0),)
    assert reserved.arc_busy("S2", "S1").intervals == ()


def test_without_occupancies_of_a_request_its_room_comes_back() -> None:
    item = PlmmOccupancyInterval("r1", 0, 8.0, start=0.0, end=20.0)
    reserved = Reservations().with_occupancies([("S1", item)])
    assert not reserved.plmm_has_room(_sat(), 4.0, 5.0, 6.0, EPS)
    freed = reserved.without_occupancies_of("r1")
    assert freed.plmm_has_room(_sat(), 4.0, 5.0, 6.0, EPS)
