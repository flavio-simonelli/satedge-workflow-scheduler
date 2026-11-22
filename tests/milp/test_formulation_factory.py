"""Tests of the formulation of each level: the blocks of a level must be the ones of the document."""

from __future__ import annotations

import itertools

import pytest

from satedge_workflow_scheduler.solvers.milp import formulation_factory as factory
from satedge_workflow_scheduler.solvers.milp.blocks.assignment import AssignmentBlock
from satedge_workflow_scheduler.solvers.milp.config import (
    Level,
    MilpConfig,
    Strengthening,
)
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.formulation_factory import (
    block_names,
    blocks_for,
)


def test_the_fixed_delay_level_has_fixed_delays_and_no_task_timing() -> None:
    names = block_names(MilpConfig(level=Level.FIXED_DELAY_WINDOWS))
    assert "fixed_delay" in names
    assert "task_timing" not in names
    assert "path" not in names


def test_from_routing_on_the_task_timing_replaces_the_fixed_delays() -> None:
    names = block_names(MilpConfig(level=Level.ROUTING))
    assert "task_timing" in names
    assert "fixed_delay" not in names


@pytest.mark.parametrize(
    ("level", "added"),
    [
        (Level.LINK_CONTENTION, "link_contention"),
        (Level.PLMM_CAPACITY, "plmm_flow"),
        (Level.ONLINE, "online"),
    ],
)
def test_each_level_adds_its_block_at_the_end_of_the_previous_one(
    level: Level, added: str
) -> None:
    names = block_names(MilpConfig(level=level))
    assert names[-1] == added


def test_from_routing_on_the_levels_only_grow() -> None:
    levels = (Level.ROUTING, Level.LINK_CONTENTION, Level.PLMM_CAPACITY, Level.ONLINE)
    for smaller, larger in itertools.pairwise(levels):
        assert set(block_names(MilpConfig(level=smaller))) <= set(
            block_names(MilpConfig(level=larger))
        )


def test_strengthenings_add_their_blocks_only_when_chosen() -> None:
    plain = block_names(MilpConfig(level=Level.ROUTING))
    strengthened = block_names(
        MilpConfig(
            level=Level.ROUTING,
            strengthening=Strengthening(
                arrival_without_big_m=True, contact_capacity=True
            ),
        )
    )
    assert strengthened[: len(plain)] == plain
    assert strengthened[len(plain) :] == ("arrival_without_big_m", "contact_capacity")


def test_a_block_without_a_class_is_a_model_error_not_an_omission() -> None:
    with pytest.raises(ModelBuildError, match="not implemented yet"):
        blocks_for(MilpConfig(level=Level.FIXED_DELAY_WINDOWS))


def test_blocks_are_instantiated_in_the_order_of_the_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # register every block of the level with a class: the order must follow the level
    for name in block_names(MilpConfig(level=Level.ROUTING)):
        monkeypatch.setitem(factory.IMPLEMENTED, name, AssignmentBlock)
    built = blocks_for(MilpConfig(level=Level.ROUTING))
    assert len(built) == len(block_names(MilpConfig(level=Level.ROUTING)))
    assert all(isinstance(b, AssignmentBlock) for b in built)


def test_the_plmm_formulation_is_chosen_in_the_config() -> None:
    from satedge_workflow_scheduler.solvers.milp.config import PlmmFormulation

    flow = block_names(MilpConfig(level=Level.PLMM_CAPACITY))
    events = block_names(
        MilpConfig(level=Level.PLMM_CAPACITY, plmm_formulation=PlmmFormulation.EVENTS)
    )
    assert "plmm_flow" in flow and "plmm_events" not in flow
    assert "plmm_events" in events and "plmm_flow" not in events
