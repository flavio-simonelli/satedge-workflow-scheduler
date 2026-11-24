"""Tests of the lazy PLMM (rule 10): the PLMM on a subset of satellites, and the lazy loop.

The instance with the PLMM of S1 at 4.5 Gb is the first one where the memory is full: without
the PLMM, the optimal calendar keeps the raw data and the output of the first task of r1 on S1
together, 5 Gb.
"""

from __future__ import annotations

import pytest
from sample import Built, build, sample_instance

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.errors import SolverError
from satedge_workflow_scheduler.solvers.milp.backends.ports import SolveParams
from satedge_workflow_scheduler.solvers.milp.blocks.link_window import LinkWindowBlock
from satedge_workflow_scheduler.solvers.milp.blocks.path import PathBlock
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
)
from satedge_workflow_scheduler.solvers.milp.config import (
    LazyGroup,
    Level,
    MilpConfig,
    PlmmFormulation,
)
from satedge_workflow_scheduler.solvers.milp.formulation_factory import (
    block_names,
    blocks_for,
)
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData
from satedge_workflow_scheduler.solvers.milp.strategy import (
    BackendFactory,
    DirectSolve,
    LazyPlmm,
    LazySolve,
    strategy_for,
)

PLMM = MilpConfig(level=Level.PLMM_CAPACITY)
LAZY = MilpConfig(level=Level.PLMM_CAPACITY, lazy=frozenset({LazyGroup.PLMM}))


def _tight() -> ProblemInstance:
    return sample_instance({"S1": 4.5})


# --- the PLMM on a subset of satellites --------------------------------------------------


def test_an_empty_set_of_satellites_leaves_the_plmm_out() -> None:
    names = block_names(PLMM, restricted={LazyGroup.PLMM: frozenset()})
    assert not {"plmm_occupation", "plmm_flow"} & set(names)
    assert {"plmm_occupation", "plmm_flow"} <= set(
        block_names(PLMM, restricted={LazyGroup.PLMM: frozenset({"S1"})})
    )


def test_the_factory_builds_the_plmm_blocks_on_the_chosen_satellites() -> None:
    data = ModelData(sample_instance(), DeadlineMode.HARD)
    blocks = blocks_for(PLMM, restricted={LazyGroup.PLMM: frozenset({"S1"})})
    occupation = next(b for b in blocks if isinstance(b, PlmmOccupationBlock))
    assert occupation.chosen(data.satellites) == ("S1",)


def test_the_plmm_variables_exist_only_on_the_chosen_satellites() -> None:
    built: Built = build(
        [
            PathBlock(),
            StoreAndForwardBlock(),
            LinkWindowBlock(),
            PlmmOccupationBlock({"S1"}),
            PlmmFlowBlock({"S1"}),
        ]
    )
    occupation = built.model.families.get(PlmmOccupation)
    flow = built.model.families.get(FlowVariables)
    assert {v for _, v in occupation.start} == {"S1"}
    assert {v for _, _, v in flow.before} == {"S1"}
    assert set(flow.unused) == {"S1"}


def test_the_configuration_chooses_the_strategy() -> None:
    assert isinstance(strategy_for(PLMM), DirectSolve)
    assert isinstance(strategy_for(LAZY), LazySolve)


# --- with Gurobi -------------------------------------------------------------------------


def _gurobi() -> BackendFactory:
    pytest.importorskip("gurobipy")
    from satedge_workflow_scheduler.solvers.milp.backends.gurobi import GurobiBackend

    return GurobiBackend


def test_without_the_plmm_the_tight_instance_overloads_s1() -> None:
    from satedge_workflow_scheduler.solvers.milp.solver import MilpSolver
    from satedge_workflow_scheduler.solvers.milp.strategy import overloaded_satellites

    instance = _tight()
    config = MilpConfig(level=Level.LINK_CONTENTION)
    solution = MilpSolver(config, _gurobi()).solve(instance)
    assert overloaded_satellites(instance, solution) == {"S1"}


def test_the_lazy_loop_adds_s1_and_reaches_the_eager_optimum() -> None:
    from satedge_workflow_scheduler.scheduling.feasibility import check

    instance = _tight()
    data = ModelData(instance, DeadlineMode.HARD)
    eager = DirectSolve().solve(instance, data, PLMM, _gurobi()).solution
    lazy = LazySolve([LazyPlmm()])
    solution = lazy.solve(instance, data, LAZY, _gurobi()).solution

    assert lazy.rounds == (
        {LazyGroup.PLMM: frozenset()},
        {LazyGroup.PLMM: frozenset({"S1"})},
    )
    assert check(instance, solution, DeadlineMode.HARD) == ()
    assert check(instance, eager, DeadlineMode.HARD) == ()
    assert eager.objective is not None and solution.objective is not None
    assert solution.objective.response == pytest.approx(eager.objective.response)
    # the full memory of S1 delays r1: delivered at 52.5 instead of 46.5
    by_id = {o.request_id: o for o in solution.outcomes}
    assert by_id["r1"].delivery_time == pytest.approx(52.5)


def test_the_two_formulations_agree_when_the_memory_is_full() -> None:
    from satedge_workflow_scheduler.scheduling.feasibility import check

    instance = _tight()
    data = ModelData(instance, DeadlineMode.HARD)
    responses = []
    for formulation in (PlmmFormulation.FLOW, PlmmFormulation.EVENTS):
        config = MilpConfig(level=Level.PLMM_CAPACITY, plmm_formulation=formulation)
        solution = DirectSolve().solve(instance, data, config, _gurobi()).solution
        assert check(instance, solution, DeadlineMode.HARD) == ()
        assert solution.objective is not None
        responses.append(solution.objective.response)
    assert responses[0] == pytest.approx(responses[1])


def test_the_lazy_loop_has_a_time_limit_for_all_its_rounds() -> None:
    instance = _tight()
    config = MilpConfig(
        level=Level.PLMM_CAPACITY,
        lazy=frozenset({LazyGroup.PLMM}),
        solve=SolveParams(time_limit=1e-9),
    )
    with pytest.raises(SolverError, match="time limit"):
        LazySolve([LazyPlmm()]).solve(
            instance, ModelData(instance, DeadlineMode.HARD), config, _gurobi()
        )
