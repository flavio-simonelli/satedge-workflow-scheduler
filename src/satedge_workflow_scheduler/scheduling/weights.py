"""Lexicographic weights of the objective: rejection costs (hard) and tardiness weights (soft)."""

from __future__ import annotations

from collections.abc import Sequence

from satedge_workflow_scheduler.domain.errors import InstanceError
from satedge_workflow_scheduler.domain.requests import Priority, Request


def rejection_weights(requests: Sequence[Request]) -> dict[str, float]:
    """Return the cost ``κ_r`` of rejecting each request, keyed by request identifier.

    A non urgent request costs 1. An urgent request costs ``|N| + 1``, where ``|N|`` is the
    number of non urgent requests, so that rejecting one urgent request costs more than
    rejecting all the non urgent ones together.
    """
    non_urgent_count = sum(1 for r in requests if r.priority is Priority.NON_URGENT)
    return {
        r.id: (non_urgent_count + 1.0) if r.priority is Priority.URGENT else 1.0
        for r in requests
    }


def tardiness_weights(
    requests: Sequence[Request], horizon: float, resolution: float = 1.0
) -> dict[str, float]:
    """Return the tardiness weight ``w_r`` of each request, keyed by request identifier.

    A non urgent request has weight 1. An urgent request has weight
    ``1 + (1 / resolution) * Σ max(0, horizon - d_q)``, summed over the non urgent requests
    ``q`` with deadline ``d_q``: the urgent lateness dominates the non urgent one.

    Raises:
        InstanceError: If ``resolution`` is not strictly positive.
    """
    if resolution <= 0:
        raise InstanceError(f"weight resolution must be > 0, got {resolution}")

    slack_of_non_urgent = sum(
        max(0.0, horizon - r.deadline)
        for r in requests
        if r.priority is Priority.NON_URGENT
    )
    return {
        r.id: (
            1.0 + slack_of_non_urgent / resolution
            if r.priority is Priority.URGENT
            else 1.0
        )
        for r in requests
    }
