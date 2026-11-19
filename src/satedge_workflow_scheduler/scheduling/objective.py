"""Objective of a schedule, evaluated for each deadline mode, with its terms kept apart."""

from __future__ import annotations

from collections.abc import Sequence

from satedge_workflow_scheduler.common.errors import SolverError
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.domain.solution import ObjectiveValue, RequestOutcome
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
) -> ObjectiveValue:
    """Return the objective of the outcomes of ``instance``, as two separate terms.

    ``primary``: in ``HARD`` mode the sum of ``κ_r`` over the rejected requests; in ``SOFT``
    mode the sum of ``w_r * tardiness_r`` over all requests.
    ``response``: the sum of ``delivery - release`` over the accepted requests. Rejected
    requests have no delivery and contribute zero.

    Raises:
        SolverError: If an outcome refers to a request that is not in the instance.
    """
    requests = {r.id: r for r in instance.requests}
    for outcome in outcomes:
        if outcome.request_id not in requests:
            raise SolverError(f"outcome for unknown request {outcome.request_id!r}")

    if mode is DeadlineMode.HARD:
        costs = rejection_weights(instance.requests)
        primary = sum(costs[o.request_id] for o in outcomes if not o.accepted)
    else:
        weights = tardiness_weights(instance.requests, instance.horizon)
        primary = sum(
            weights[o.request_id] * tardiness(o, requests[o.request_id])
            for o in outcomes
        )

    response = sum(
        o.delivery_time - requests[o.request_id].release
        for o in outcomes
        if o.accepted and o.delivery_time is not None
    )
    return ObjectiveValue(primary=float(primary), response=float(response))
