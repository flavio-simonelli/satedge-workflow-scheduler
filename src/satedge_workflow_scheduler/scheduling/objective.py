"""Objective of a schedule, evaluated for each deadline mode."""

from __future__ import annotations

from collections.abc import Sequence

from satedge_workflow_scheduler.common.errors import InstanceError
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.domain.solution import RequestOutcome
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.weights import (
    rejection_weights,
    tardiness_weights,
)


def tardiness(outcome: RequestOutcome, request: Request) -> float:
    """Return ``max(0, delivery - deadline)`` of an accepted request, 0 if it is rejected."""
    if not outcome.accepted or outcome.delivery_time is None:
        return 0.0
    return max(0.0, outcome.delivery_time - request.deadline)


def evaluate(
    instance: ProblemInstance, outcomes: Sequence[RequestOutcome], mode: DeadlineMode
) -> float:
    """Return the objective value of the outcomes of ``instance``.

    ``HARD``: the sum of ``κ_r`` over the rejected requests (to be minimised).
    ``SOFT``: the sum of ``w_r * tardiness_r`` over all requests (to be minimised).

    Raises:
        InstanceError: If an outcome refers to a request that is not in the instance.
    """
    requests = {r.id: r for r in instance.requests}
    for outcome in outcomes:
        if outcome.request_id not in requests:
            raise InstanceError(f"unknown request {outcome.request_id!r}")

    if mode is DeadlineMode.HARD:
        costs = rejection_weights(instance.requests)
        return sum(costs[o.request_id] for o in outcomes if not o.accepted)

    weights = tardiness_weights(instance.requests, instance.horizon)
    return sum(
        weights[o.request_id] * tardiness(o, requests[o.request_id]) for o in outcomes
    )
