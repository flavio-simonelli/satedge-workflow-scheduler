"""Tests of the exact reductions (rules 1–5 of the document): what they remove, and that they
change nothing in the optimum.

The unit tests read the sets of :class:`Reductions` and the model the builder writes with them, on
small instances whose intervals are easy to compute by hand. The tests with Gurobi solve each
instance with and without the reductions and compare the two optima, checked by the scheduling
module, which does not know the model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import pytest
from sample import online_instance, sample_instance

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
from satedge_workflow_scheduler.domain.solution import Schedule, Transfer
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.feasibility import check
from satedge_workflow_scheduler.solvers.errors import InfeasibleModelError
from satedge_workflow_scheduler.solvers.milp.backends.fake_recording import (
    FakeRecordingBackend,
)
from satedge_workflow_scheduler.solvers.milp.blocks.path import RoutingVariables
from satedge_workflow_scheduler.solvers.milp.blocks.plmm_flow import FlowVariables
from satedge_workflow_scheduler.solvers.milp.builder import BuiltModel, ModelBuilder
from satedge_workflow_scheduler.solvers.milp.config import (
    LazyGroup,
    Level,
    MilpConfig,
    PlmmFormulation,
)
from satedge_workflow_scheduler.solvers.milp.encoding import encode
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.formulation_factory import blocks_for
from satedge_workflow_scheduler.solvers.milp.model_data import (
    DataKey,
    ModelData,
    TaskKey,
)
from satedge_workflow_scheduler.solvers.milp.reductions import (
    Reductions,
    reductions_for,
)

HARD, SOFT = DeadlineMode.HARD, DeadlineMode.SOFT
PLMM = MilpConfig(level=Level.PLMM_CAPACITY)


def _instance(
    *,
    windows: tuple[TimeWindow, ...] = (TimeWindow(0.0, 100.0),),
    capacity: Mapping[str, float] | None = None,
    r1_deadline: float = 500.0,
    r2_release: float = 0.0,
    r2_execution: float = 5.0,
) -> ProblemInstance:
    """The network of the sample instance, with the windows, capacities and requests given.

    S1 -> S2 and S2 -> S1 always available, S2 -> G in [0, 1000]. HPC at 1 Gbps on both.
    r1: raw 2 Gb on S1 from 0, tasks of 10 s (output 3 Gb) and 20 s (output 1 Gb).
    r2: raw 1 Gb on S2, one task (output 2 Gb), deadline 550.
    """
    hpc = WindowSet(windows)
    sizes = {"S1": 10.0, "S2": 10.0, **(capacity or {})}
    s1 = Satellite(id="S1", plmm_capacity=sizes["S1"], hpc_rate=1.0, hpc_windows=hpc)
    s2 = Satellite(id="S2", plmm_capacity=sizes["S2"], hpc_rate=1.0, hpc_windows=hpc)
    contact = WindowedAvailability(WindowSet((TimeWindow(0.0, 1000.0),)))
    network = Network(
        planes=(Plane(id=1, satellites=(s1, s2), com="S2"),),
        ground_station=GroundStation(id="G"),
        arcs=(
            Arc("S1", "S2", capacity=1.0, latency=1.0, availability=AlwaysAvailable()),
            Arc("S2", "S1", capacity=1.0, latency=1.0, availability=AlwaysAvailable()),
            Arc("S2", "G", capacity=2.0, latency=5.0, availability=contact),
        ),
    )
    r1 = Request(
        id="r1",
        priority=Priority.URGENT,
        origin="S1",
        release=0.0,
        deadline=r1_deadline,
        raw_size=2.0,
        tasks=(
            Task(execution_time=10.0, output_size=3.0),
            Task(execution_time=20.0, output_size=1.0),
        ),
    )
    r2 = Request(
        id="r2",
        priority=Priority.NON_URGENT,
        origin="S2",
        release=r2_release,
        deadline=550.0,
        raw_size=1.0,
        tasks=(Task(execution_time=r2_execution, output_size=2.0),),
    )
    return ProblemInstance(
        network=network,
        requests=(r1, r2),
        horizon=600.0,
        plan_release={"S1": 0.0, "S2": 0.0},
    )


def _apart(capacity: float = 4.0) -> ProblemInstance:
    """r1 must end by 150, in the window [0, 100]; r2 is released at 200, so it runs in the
    window [200, 300]. The two requests never share an HPC, an arc or a PLMM at the same time.
    The PLMM of S1 holds 4 Gb, less than the 2 + 3 Gb that r1 alone may keep there."""
    return _instance(
        windows=(TimeWindow(0.0, 100.0), TimeWindow(200.0, 300.0)),
        capacity={"S1": capacity},
        r1_deadline=150.0,
        r2_release=200.0,
    )


def _build(
    instance: ProblemInstance,
    reductions: bool,
    config: MilpConfig = PLMM,
    mode: DeadlineMode = HARD,
) -> tuple[ModelData, BuiltModel, FakeRecordingBackend]:
    data = ModelData(instance, mode)
    backend = FakeRecordingBackend()
    blocks = blocks_for(config, online=data.is_online)
    model = ModelBuilder(data, backend, reductions_for(data, reductions)).build(blocks)
    return data, model, backend


def _names(backend: FakeRecordingBackend) -> set[str]:
    return {v.name for v in backend.variables}


# --- without reductions ------------------------------------------------------------------


def test_without_reductions_every_key_is_in_the_model() -> None:
    data = ModelData(sample_instance(), HARD)
    none = Reductions.none(data)
    assert none.excluded == frozenset()
    assert none.task_pairs == data.task_pairs
    assert len(none.link_triples) == len(data.link_pairs) * len(data.arc_ids)
    assert none.plmm_satellites == data.satellites
    assert all(none.arcs(a) == data.arc_ids for a in data.data)


def test_the_builder_without_reductions_writes_the_model_as_before() -> None:
    data = ModelData(sample_instance(), HARD)
    blocks = blocks_for(PLMM)
    default, explicit = FakeRecordingBackend(), FakeRecordingBackend()
    ModelBuilder(data, default).build(blocks)
    ModelBuilder(data, explicit, Reductions.none(data)).build(blocks_for(PLMM))
    assert default.variables == explicit.variables
    assert default.rows == explicit.rows


def test_the_reductions_shrink_the_model() -> None:
    _, _, plain = _build(sample_instance(), reductions=False)
    _, _, reduced = _build(sample_instance(), reductions=True)
    assert len(reduced.variables) < len(plain.variables)
    assert len(reduced.rows) < len(plain.rows)


# --- rule 1: satellites and computing windows ---------------------------------------------


def test_a_window_after_every_deadline_has_no_u() -> None:
    # the window [560, 590] starts after both deadlines: no task can use it
    instance = _instance(windows=(TimeWindow(0.0, 100.0), TimeWindow(560.0, 590.0)))
    data, model, _ = _build(instance, reductions=True)
    task = TaskKey("r1", 0)
    assert model.reductions.task_windows(task, "S1") == (0,)
    assert model.variables.windows(task, "S1") == (0,)
    with pytest.raises(ModelBuildError):
        model.variables.u(task, "S1", 1)
    assert Reductions.none(data).task_windows(task, "S1") == (0, 1)


# --- rule 2: pairs of tasks ---------------------------------------------------------------


def test_tasks_in_windows_that_never_meet_leave_the_pairs() -> None:
    data, model, backend = _build(_apart(), reductions=True)
    assert data.task_pairs  # the pairs exist in the data...
    assert model.reductions.task_pairs == ()  # ...but none can overlap
    assert not any(n.startswith(("same[", "order[")) for n in _names(backend))
    assert model.rows["task overlap"] == 0


def test_tasks_that_may_meet_keep_only_the_shared_windows() -> None:
    data = ModelData(sample_instance(), HARD)
    reductions = Reductions.exact(data)
    pair = (TaskKey("r1", 0), TaskKey("r2", 0))
    assert reductions.shared_windows(*pair) == (("S1", 0), ("S2", 0))


# --- rule 3: arcs and visibility windows --------------------------------------------------


def test_an_intermediate_data_never_crosses_the_arc_into_the_ground() -> None:
    _, model, backend = _build(sample_instance(), reductions=True)
    raw, ground_arc = DataKey("r1", 0), 2
    routing = model.families.get(RoutingVariables)
    assert not model.reductions.crosses(raw, ground_arc)
    assert routing.chi(raw, ground_arc).terms == {}
    assert f"nu[{raw},{ground_arc},0]" not in _names(backend)
    assert f"xi[{raw},{ground_arc}]" not in _names(backend)
    # the final output keeps the arc
    assert model.reductions.crosses(DataKey("r1", 2), ground_arc)


def test_a_calendar_on_a_removed_arc_is_not_a_calendar_of_the_model() -> None:
    data, model, _ = _build(sample_instance(), reductions=True)
    bad = Schedule((), (Transfer("r1", 0, "S2", "G", 10.0, 11.0),))
    with pytest.raises(ModelBuildError, match="removed"):
        encode(data, model, bad)


# --- rule 4: pairs of data on the links ---------------------------------------------------


def test_transmissions_that_never_meet_have_no_order() -> None:
    _, model, backend = _build(_apart(), reductions=True)
    assert model.reductions.link_triples == ()
    assert not any(n.startswith("sigma[") for n in _names(backend))


def test_transmissions_that_may_meet_keep_their_order() -> None:
    data = ModelData(sample_instance(), HARD)
    triples = Reductions.exact(data).link_triples
    assert (DataKey("r1", 0), DataKey("r2", 0), 0) in triples
    # the raw data of r1 never crosses S2 -> G, so it meets nothing there
    assert all(not (a == DataKey("r1", 0) and e == 2) for a, _, e in triples)


# --- rule 5: PLMM -------------------------------------------------------------------------


def test_a_plmm_that_can_never_fill_has_no_capacity_constraint() -> None:
    # 10 Gb on each satellite: more than all the data of both requests together
    _, model, backend = _build(sample_instance(), reductions=True)
    assert model.reductions.plmm_satellites == ()
    assert model.rows["plmm occupation"] == 0
    assert model.rows["plmm flow"] == 0
    assert not any(n.startswith(("pi[", "occs[")) for n in _names(backend))


def test_only_the_plmm_that_can_fill_keeps_its_capacity() -> None:
    _, model, _ = _build(sample_instance({"S1": 4.0}), reductions=True)
    assert model.reductions.plmm_satellites == ("S1",)


def test_stays_that_never_overlap_have_a_fixed_order() -> None:
    _, model, backend = _build(_apart(), reductions=True)
    early, late = DataKey("r1", 0), DataKey("r2", 0)
    assert model.reductions.plmm_satellites == ("S1",)
    assert model.reductions.plmm_order(early, late, "S1") is True
    assert model.reductions.plmm_order(late, early, "S1") is False
    flow = model.families.get(FlowVariables)
    assert flow.pi(early, late, "S1").constant == 1.0
    assert flow.pi(late, early, "S1").constant == 0.0
    # π fixed: no variable, no flow back, and neither (22) nor (23) for the pair
    names = _names(backend)
    assert f"pi[{early},{late},S1]" not in names
    assert f"f[{early},{late},S1]" in names
    assert f"f[{late},{early},S1]" not in names
    rows = {r.name for r in backend.rows}
    assert f"order[{early},{late},S1]" not in rows
    assert f"after-leaving[{early},{late},S1]" not in rows


def test_the_events_of_a_plmm_are_as_many_as_the_data_it_can_hold() -> None:
    config = replace(PLMM, plmm_formulation=PlmmFormulation.EVENTS)
    instance = online_instance({"S1": 0.0, "S2": 10.0})
    instance = replace(
        instance,
        network=_instance(capacity={"S1": 1.0, "S2": 1.0}).network,
    )
    data, model, backend = _build(instance, reductions=True, config=config)
    # the committed r2 never reaches S1: only the three data of r1 can be there
    assert model.reductions.plmm_data("S1") == data.data_of("r1")
    events = [n for n in _names(backend) if n.startswith("tau[S1,")]
    assert len(events) == 3


# --- committed requests: their calendar is their bounds -------------------------------------


def test_a_committed_request_keeps_only_its_calendar() -> None:
    _, model, _ = _build(online_instance({"S1": 0.0, "S2": 0.0}), reductions=True)
    reductions = model.reductions
    task = TaskKey("r2", 0)
    assert reductions.task_windows(task, "S2") == (0,)
    assert reductions.task_windows(task, "S1") == ()
    assert reductions.arcs(DataKey("r2", 0)) == ()
    assert reductions.arcs(DataKey("r2", 1)) == (2,)


# --- requests that can never be served ------------------------------------------------------


def _impossible() -> ProblemInstance:
    # the task of r2 lasts 1 + 200 + 2 s, longer than the window [0, 100]
    return _instance(r2_execution=200.0)


def test_a_request_that_can_never_be_served_is_rejected_a_priori() -> None:
    _, model, backend = _build(_impossible(), reductions=True)
    assert model.reductions.excluded == frozenset({"r2"})
    assert model.variables.rejection("r2").constant == 1.0
    assert "rho[r2]" not in _names(backend)
    assert model.variables.windows(TaskKey("r2", 0), "S2") == ()


def test_a_window_short_by_less_than_the_checker_tolerance_is_kept() -> None:
    # the task of r2 lasts 1 + 97 + 2 s and 5e-7 more: it overflows [0, 100] by less than the
    # checker accepts, so a calendar that runs it there is feasible and must stay in the model
    data = ModelData(_instance(r2_execution=97.0 + 5e-7), HARD)
    reductions = Reductions.exact(data)
    assert reductions.excluded == frozenset()
    assert reductions.task_windows(TaskKey("r2", 0), "S2") == (0,)


def test_under_soft_deadlines_it_makes_the_instance_infeasible() -> None:
    data = ModelData(_impossible(), SOFT)
    with pytest.raises(InfeasibleModelError) as error:
        Reductions.exact(data)
    assert set(error.value.reasons) == {"r2"}
    assert "non entra in nessuna finestra di calcolo" in error.value.reasons["r2"]


# --- the optimum does not change (Gurobi) -------------------------------------------------


_INSTANCES: dict[str, Callable[[], ProblemInstance]] = {
    "sample": sample_instance,
    "tight": lambda: sample_instance({"S1": 4.0}),
    "apart": _apart,
    "online": lambda: online_instance({"S1": 0.0, "S2": 10.0}),
}

_CONFIGS = {
    "fixed delay": MilpConfig(level=Level.FIXED_DELAY_WINDOWS),
    "routing": MilpConfig(level=Level.ROUTING),
    "contention": MilpConfig(level=Level.LINK_CONTENTION),
    "plmm flow": PLMM,
    "plmm events": replace(PLMM, plmm_formulation=PlmmFormulation.EVENTS),
    "plmm lazy": replace(PLMM, lazy=frozenset({LazyGroup.PLMM})),
}


@pytest.mark.parametrize("mode", [HARD, SOFT])
@pytest.mark.parametrize("config_name", list(_CONFIGS))
@pytest.mark.parametrize("instance_name", list(_INSTANCES))
def test_the_reductions_keep_the_optimum(
    instance_name: str, config_name: str, mode: DeadlineMode
) -> None:
    pytest.importorskip("gurobipy")
    from satedge_workflow_scheduler.solvers.milp.backends.gurobi import GurobiBackend
    from satedge_workflow_scheduler.solvers.milp.solver import MilpSolver

    instance = _INSTANCES[instance_name]()
    objectives = []
    for reductions in (False, True):
        config = replace(_CONFIGS[config_name], deadline=mode, reductions=reductions)
        solution = MilpSolver(config, GurobiBackend).solve(instance)
        if config.level.plmm:
            # the lower levels relax some constraints of the checker on purpose
            assert check(instance, solution, mode) == ()
        assert solution.objective is not None
        objectives.append(solution.objective)
    plain, reduced = objectives
    assert reduced.primary == pytest.approx(plain.primary)
    assert reduced.response == pytest.approx(plain.response)


def test_an_impossible_request_is_rejected_with_or_without_the_reductions() -> None:
    pytest.importorskip("gurobipy")
    from satedge_workflow_scheduler.solvers.milp.backends.gurobi import GurobiBackend
    from satedge_workflow_scheduler.solvers.milp.solver import MilpSolver

    instance = _impossible()
    objectives = []
    for reductions in (False, True):
        config = replace(PLMM, reductions=reductions)
        solution = MilpSolver(config, GurobiBackend).solve(instance)
        assert not {o.request_id: o for o in solution.outcomes}["r2"].accepted
        assert check(instance, solution, HARD) == ()
        assert solution.objective is not None
        objectives.append(solution.objective.primary)
    assert objectives[1] == pytest.approx(objectives[0])
