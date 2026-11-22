"""Shared sample instance of the MILP tests: two satellites, a ground station, two requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

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
from satedge_workflow_scheduler.domain.solution import Schedule, TaskExecution, Transfer
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.fake_recording import (
    FakeRecordingBackend,
)
from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.builder import BuiltModel, ModelBuilder
from satedge_workflow_scheduler.solvers.milp.expr import Row
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData

HORIZON = 600.0


def sample_instance() -> ProblemInstance:
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


@dataclass(frozen=True)
class Built:
    """A model built from some blocks, with the context to read its handles and rows."""

    ctx: BuildContext
    backend: FakeRecordingBackend
    model: BuiltModel

    def row(self, name: str) -> Row:
        (row,) = [r for r in self.backend.rows if r.name == name]
        return row


def online_instance(plan_release: Mapping[str, float]) -> ProblemInstance:
    """The sample instance at a later epoch: r2 is committed, r1 is new.

    The committed calendar of r2 is the one the offline model finds: its task on S2 in [0, 8],
    and its output on S2 → G from 8, with a transmission of 1 s.
    """
    committed = Schedule(
        executions=(TaskExecution("r2", 0, "S2", 0.0, 8.0),),
        transfers=(Transfer("r2", 1, "S2", "G", 8.0, 9.0),),
    )
    return replace(sample_instance(), plan_release=plan_release, committed=committed)


def build(
    blocks: Sequence[ConstraintBlock[Any]],
    mode: DeadlineMode = DeadlineMode.HARD,
    instance: ProblemInstance | None = None,
) -> Built:
    """Build ``blocks`` on ``instance`` (the sample one by default), in order, through the
    builder."""
    data = ModelData(instance if instance is not None else sample_instance(), mode)
    backend = FakeRecordingBackend()
    model = ModelBuilder(data, backend).build(blocks)
    return Built(BuildContext(data, model.variables, model.families), backend, model)
