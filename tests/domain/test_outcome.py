from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.errors import InstanceError
from satedge_workflow_scheduler.domain.solution import RequestOutcome


def test_rejected_outcome_may_carry_a_reason() -> None:
    outcome = RequestOutcome("r", accepted=False, rejection_reason="no placement")
    assert outcome.rejection_reason == "no placement"


def test_accepted_outcome_must_not_carry_a_reason() -> None:
    with pytest.raises(InstanceError):
        RequestOutcome("r", accepted=True, delivery_time=5.0, rejection_reason="x")
