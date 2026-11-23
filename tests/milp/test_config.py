"""Tests of the configuration: it must accept the valid models and reject the others."""

from __future__ import annotations

import dataclasses

import pytest

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.ports import SolveParams
from satedge_workflow_scheduler.solvers.milp.config import (
    LazyGroup,
    Level,
    MilpConfig,
    Strengthening,
)
from satedge_workflow_scheduler.solvers.milp.errors import MilpConfigError


@pytest.mark.parametrize(
    ("level", "routed", "contention", "plmm"),
    [
        (Level.FIXED_DELAY_WINDOWS, False, False, False),
        (Level.ROUTING, True, False, False),
        (Level.LINK_CONTENTION, True, True, False),
        (Level.PLMM_CAPACITY, True, True, True),
    ],
)
def test_each_level_adds_exactly_its_part(
    level: Level, routed: bool, contention: bool, plmm: bool
) -> None:
    assert (level.routed, level.contention, level.plmm) == (routed, contention, plmm)


def test_levels_are_ordered_and_start_at_three() -> None:
    assert [int(level) for level in Level] == [3, 4, 5, 6]


def test_default_configuration_is_the_plain_hard_model() -> None:
    config = MilpConfig(level=Level.FIXED_DELAY_WINDOWS)
    assert config.deadline is DeadlineMode.HARD
    assert config.reductions is True
    assert config.lazy == frozenset()
    assert config.polish is True
    assert config.strengthening == Strengthening()


def test_a_configuration_is_immutable() -> None:
    config = MilpConfig(level=Level.ROUTING)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.polish = True  # type: ignore[misc]


def test_strengthenings_need_routed_communication() -> None:
    with pytest.raises(MilpConfigError, match="routed"):
        MilpConfig(
            level=Level.FIXED_DELAY_WINDOWS,
            strengthening=Strengthening(contact_capacity=True),
        )
    with pytest.raises(MilpConfigError, match="routed"):
        MilpConfig(
            level=Level.FIXED_DELAY_WINDOWS,
            strengthening=Strengthening(arrival_without_big_m=True),
        )


def test_strengthenings_are_accepted_from_level_four() -> None:
    config = MilpConfig(
        level=Level.ROUTING,
        strengthening=Strengthening(arrival_without_big_m=True, contact_capacity=True),
    )
    assert config.strengthening.contact_capacity


def test_lazy_plmm_needs_the_plmm_level() -> None:
    with pytest.raises(MilpConfigError, match="level 6"):
        MilpConfig(level=Level.LINK_CONTENTION, lazy=frozenset({LazyGroup.PLMM}))
    config = MilpConfig(level=Level.PLMM_CAPACITY, lazy=frozenset({LazyGroup.PLMM}))
    assert config.lazy == {LazyGroup.PLMM}


@pytest.mark.parametrize(
    "solve",
    [
        SolveParams(time_limit=0.0),
        SolveParams(time_limit=-1.0),
        SolveParams(mip_gap=-0.1),
        SolveParams(memory_limit_gb=0.0),
    ],
)
def test_nonsensical_solver_limits_are_rejected(solve: SolveParams) -> None:
    with pytest.raises(MilpConfigError):
        MilpConfig(level=Level.FIXED_DELAY_WINDOWS, solve=solve)


def test_solver_limits_left_unset_are_accepted() -> None:
    config = MilpConfig(
        level=Level.FIXED_DELAY_WINDOWS, solve=SolveParams(time_limit=None)
    )
    assert config.solve.time_limit is None
