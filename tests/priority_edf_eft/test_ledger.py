from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.errors import SolverError
from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Network,
    Plane,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, Satellite
from satedge_workflow_scheduler.scheduling.plmm_occupancy import PlmmOccupancyInterval
from satedge_workflow_scheduler.solvers.priority_edf_eft.ledger import (
    ArcKey,
    HpcKey,
    ResourceLedger,
)


def _sat(sid: str) -> Satellite:
    return Satellite(
        id=sid,
        plmm_capacity=10.0,
        hpc_write_rate=1.0,
        hpc_windows=(TimeWindow(0.0, 1000.0),),
    )


def _ledger() -> ResourceLedger:
    s1, s2 = _sat("S1"), _sat("S2")
    network = Network(
        planes=(
            Plane(id=1, satellites=(s1,), com="S1"),
            Plane(id=2, satellites=(s2,), com="S2"),
        ),
        ground_station=GroundStation(id="G"),
        arcs=(
            Arc(
                source="S1",
                target="S2",
                capacity=1.0,
                latency=1.0,
                availability=AlwaysAvailable(),
            ),
        ),
    )
    return ResourceLedger(network)


def _occ(size: float = 2.0) -> PlmmOccupancyInterval:
    return PlmmOccupancyInterval("r", 0, size, start=0.0, end=5.0)


def test_reservation_without_commit_is_discarded() -> None:
    ledger = _ledger()
    with ledger.transaction() as tx:
        tx.reserve(HpcKey("S1"), TimeWindow(0.0, 10.0))
        assert ledger.timeline(HpcKey("S1")).windows == (TimeWindow(0.0, 10.0),)
    assert ledger.timeline(HpcKey("S1")).windows == ()


def test_committed_reservation_stays() -> None:
    ledger = _ledger()
    with ledger.transaction() as tx:
        tx.reserve(HpcKey("S1"), TimeWindow(0.0, 10.0))
        tx.commit()
    assert ledger.timeline(HpcKey("S1")).windows == (TimeWindow(0.0, 10.0),)


def test_exception_rolls_back_the_transaction() -> None:
    ledger = _ledger()
    with pytest.raises(RuntimeError, match="boom"), ledger.transaction() as tx:
        tx.reserve(ArcKey("S1", "S2"), TimeWindow(0.0, 3.0))
        tx.add_occupancies("S1", [_occ()])
        raise RuntimeError("boom")
    assert ledger.timeline(ArcKey("S1", "S2")).windows == ()
    assert ledger.plmm("S1") == ()


def test_plmm_occupancies_are_committed_and_rolled_back_like_reservations() -> None:
    ledger = _ledger()
    with ledger.transaction() as tx:
        tx.add_occupancies("S1", [_occ()])
        assert len(ledger.plmm("S1")) == 1
    assert ledger.plmm("S1") == ()

    with ledger.transaction() as tx:
        tx.add_occupancies("S1", [_occ()])
        tx.commit()
    assert ledger.plmm("S1") == (_occ(),)


def test_committed_intervals_are_not_counted_twice_after_commit() -> None:
    ledger = _ledger()
    with ledger.transaction() as tx:
        tx.reserve(HpcKey("S1"), TimeWindow(0.0, 10.0))
        tx.commit()
        assert ledger.timeline(HpcKey("S1")).windows == (TimeWindow(0.0, 10.0),)


def test_nested_transaction_is_an_error() -> None:
    ledger = _ledger()
    with ledger.transaction(), pytest.raises(RuntimeError), ledger.transaction():
        pass


def test_reserving_after_commit_is_an_error() -> None:
    ledger = _ledger()
    with ledger.transaction() as tx:
        tx.commit()
        with pytest.raises(RuntimeError):
            tx.reserve(HpcKey("S1"), TimeWindow(0.0, 1.0))


def test_unknown_resource_is_an_error() -> None:
    ledger = _ledger()
    with pytest.raises(SolverError):
        ledger.timeline(HpcKey("GHOST"))
    with pytest.raises(SolverError):
        ledger.timeline(ArcKey("S2", "S1"))


def test_available_from_is_zero_for_now() -> None:
    assert _ledger().available_from("S1") == 0.0


def test_writing_after_the_block_has_ended_is_an_error_and_changes_nothing() -> None:
    ledger = _ledger()
    with ledger.transaction() as tx:
        pass
    with pytest.raises(RuntimeError):
        tx.reserve(HpcKey("S1"), TimeWindow(0.0, 10.0))
    with pytest.raises(RuntimeError):
        tx.commit()
    assert ledger.timeline(HpcKey("S1")).windows == ()
