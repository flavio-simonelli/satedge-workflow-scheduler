from __future__ import annotations

from collections.abc import Callable

import pytest

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.solution import Solution
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.feasibility import check


def assert_feasible(
    instance: ProblemInstance, solution: Solution, mode: DeadlineMode
) -> None:
    """Fail with every violation of the model found in ``solution``, one per line."""
    violations = check(instance, solution, mode)
    assert not violations, "\n".join(str(v) for v in violations)


@pytest.fixture
def check_valid() -> Callable[[ProblemInstance, Solution, DeadlineMode], None]:
    """Fixture giving tests access to :func:`assert_feasible`."""
    return assert_feasible
