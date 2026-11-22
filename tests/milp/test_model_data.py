"""Tests of the model data: the notation must read the instance exactly."""

from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.intervals import TimeWindow, WindowSet
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Network,
    Plane,
    WindowedAvailability,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, Satellite
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode, may_reject
from satedge_workflow_scheduler.solvers.milp.model_data import (
    DataKey,
    ModelData,
    TaskKey,
)

HORIZON = 600.0


def _instance() -> ProblemInstance:
    """Two satellites, S1 with the raw data, and the ground station G.

    S1 -> S2 is always available; S2 -> G is available only in [0, 1000]. Both satellites
    compute in [0, 100], with an HPC rate of 1 Gbps.
    """
    windows = WindowSet((TimeWindow(0.0, 100.0),))
    s1 = Satellite(id="S1", plmm_capacity=10.0, hpc_rate=1.0, hpc_windows=windows)
    s2 = Satellite(id="S2", plmm_capacity=10.0, hpc_rate=1.0, hpc_windows=windows)
    arcs = (
        Arc("S1", "S2", capacity=1.0, latency=1.0, availability=AlwaysAvailable()),
        Arc("S2", "S1", capacity=1.0, latency=1.0, availability=AlwaysAvailable()),
        Arc(
            "S2",
            "G",
            capacity=2.0,
            latency=5.0,
            availability=WindowedAvailability(WindowSet((TimeWindow(0.0, 1000.0),))),
        ),
    )
    network = Network(
        planes=(Plane(id=1, satellites=(s1, s2), com="S2"),),
        ground_station=GroundStation(id="G"),
        arcs=arcs,
    )
    request = Request(
        id="r1",
        priority=Priority.URGENT,
        origin="S1",
        release=0.0,
        deadline=500.0,
        raw_size=2.0,
        tasks=(
            Task(execution_time=10.0, output_size=3.0),
            Task(execution_time=20.0, output_size=1.0),
        ),
    )
    other = Request(
        id="r2",
        priority=Priority.NON_URGENT,
        origin="S2",
        release=0.0,
        deadline=550.0,
        raw_size=1.0,
        tasks=(Task(execution_time=5.0, output_size=2.0),),
    )
    return ProblemInstance(
        network=network,
        requests=(request, other),
        horizon=HORIZON,
        plan_release={"S1": 0.0, "S2": 0.0},
    )


@pytest.fixture
def data() -> ModelData:
    return ModelData(_instance(), DeadlineMode.HARD)


# --- sets ----------------------------------------------------------------------------


def test_tasks_follow_the_pipelines_in_order(data: ModelData) -> None:
    assert data.tasks == (
        TaskKey("r1", 0),
        TaskKey("r1", 1),
        TaskKey("r2", 0),
    )


def test_each_pipeline_has_one_datum_more_than_its_tasks(data: ModelData) -> None:
    assert data.data == (
        DataKey("r1", 0),
        DataKey("r1", 1),
        DataKey("r1", 2),
        DataKey("r2", 0),
        DataKey("r2", 1),
    )


def test_nodes_are_the_satellites_and_the_ground_station(data: ModelData) -> None:
    assert data.satellites == ("S1", "S2")
    assert data.nodes == ("S1", "S2", "G")
    assert data.ground_station == "G"


# --- structure of the pipelines ------------------------------------------------------


def test_raw_data_has_no_producer_and_is_consumed_by_the_first_task(
    data: ModelData,
) -> None:
    raw = DataKey("r1", 0)
    assert data.producer(raw) is None
    assert data.consumer(raw) == TaskKey("r1", 0)


def test_intermediate_data_links_consecutive_tasks(data: ModelData) -> None:
    middle = DataKey("r1", 1)
    assert data.producer(middle) == TaskKey("r1", 0)
    assert data.consumer(middle) == TaskKey("r1", 1)


def test_final_output_is_produced_by_the_last_task_and_goes_to_ground(
    data: ModelData,
) -> None:
    final = DataKey("r1", 2)
    assert data.producer(final) == TaskKey("r1", 1)
    assert data.consumer(final) is None
    assert data.is_final(final)
    assert not data.is_final(DataKey("r1", 1))


def test_origin_is_the_satellite_of_the_raw_data(data: ModelData) -> None:
    assert data.origin("r1") == "S1"
    assert data.origin("r2") == "S2"


# --- parameters ----------------------------------------------------------------------


def test_duration_is_read_input_execution_and_write(data: ModelData) -> None:
    # raw 2 Gb read at 1 Gbps (2 s), execution 10 s, output 3 Gb written (3 s)
    assert data.duration(TaskKey("r1", 0), "S1") == pytest.approx(15.0)
    # second task: its input is the 3 Gb intermediate datum, output 1 Gb
    assert data.duration(TaskKey("r1", 1), "S2") == pytest.approx(3.0 + 20.0 + 1.0)


def test_sizes_are_the_sizes_of_the_pipeline(data: ModelData) -> None:
    assert data.size(DataKey("r1", 0)) == pytest.approx(2.0)
    assert data.size(DataKey("r1", 1)) == pytest.approx(3.0)
    assert data.size(DataKey("r1", 2)) == pytest.approx(1.0)


def test_transmission_time_and_latency_come_from_the_arc(data: ModelData) -> None:
    arc_to_ground = next(e for e in data.arc_ids if data.endpoints(e)[1] == "G")
    assert data.transmission_time(DataKey("r1", 2), arc_to_ground) == pytest.approx(0.5)
    assert data.latency(arc_to_ground) == pytest.approx(5.0)


def test_always_available_arcs_have_no_windows(data: ModelData) -> None:
    always = next(e for e in data.arc_ids if data.endpoints(e)[0] == "S1")
    assert data.link_windows(always) is None


def test_windowed_arcs_expose_their_windows(data: ModelData) -> None:
    windowed = next(e for e in data.arc_ids if data.endpoints(e)[1] == "G")
    assert data.link_windows(windowed) == WindowSet((TimeWindow(0.0, 1000.0),))


def test_adjacency_matches_the_network(data: ModelData) -> None:
    into_ground = data.incoming("G")
    assert [data.endpoints(e)[0] for e in into_ground] == ["S2"]
    assert data.outgoing("G") == ()


def test_plan_release_and_horizon_are_read(data: ModelData) -> None:
    assert data.plan_release("S2") == 0.0
    assert data.horizon == HORIZON


# --- deadlines and costs -------------------------------------------------------------


def test_hard_deadlines_use_each_request_deadline() -> None:
    data = ModelData(_instance(), DeadlineMode.HARD)
    assert data.limit("r1") == pytest.approx(500.0)
    assert data.limit("r2") == pytest.approx(550.0)
    assert data.may_reject()


def test_soft_deadlines_use_the_horizon_and_forbid_rejection() -> None:
    data = ModelData(_instance(), DeadlineMode.SOFT)
    assert data.limit("r1") == pytest.approx(HORIZON)
    assert data.limit("r2") == pytest.approx(HORIZON)
    assert not data.may_reject()


def test_costs_follow_the_priority_classes(data: ModelData) -> None:
    # one non urgent request: it costs 1; the urgent one costs |N| + 1 = 2
    assert data.rejection_costs == {"r1": 2.0, "r2": 1.0}


def test_model_data_holds_only_new_requests(data: ModelData) -> None:
    assert list(data.request_ids) == ["r1", "r2"]
    assert data.origin("r2") == "S2"


def test_may_reject_follows_the_mode() -> None:
    assert may_reject(DeadlineMode.HARD)
    assert not may_reject(DeadlineMode.SOFT)
