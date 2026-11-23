from __future__ import annotations

import pytest

from satedge_workflow_scheduler.solvers.sequential.config import (
    Ordering,
    Placement,
    Routing,
    SequentialConfig,
)
from satedge_workflow_scheduler.solvers.sequential.errors import SequentialConfigError
from satedge_workflow_scheduler.solvers.sequential.factory import router_for
from satedge_workflow_scheduler.solvers.sequential.routing.safe_interval import (
    SafeIntervalRouter,
)


def test_the_default_is_priority_edf_with_eft_and_safe_intervals() -> None:
    config = SequentialConfig()
    assert config.ordering is Ordering.PRIORITY_EDF
    assert config.placement is Placement.EFT
    assert config.routing is Routing.SAFE_INTERVAL


def test_no_backtracks_is_the_greedy_placement() -> None:
    assert SequentialConfig(max_backtracks=0).max_backtracks == 0


def test_negative_backtracks_are_refused() -> None:
    with pytest.raises(SequentialConfigError):
        SequentialConfig(max_backtracks=-1)


def test_a_tolerance_must_be_positive() -> None:
    with pytest.raises(SequentialConfigError):
        SequentialConfig(eps=0.0)


def test_the_factory_gives_the_router_of_the_configuration() -> None:
    assert isinstance(router_for(Routing.SAFE_INTERVAL, 1e-9), SafeIntervalRouter)
