"""Objective of a schedule: lexicographic weights, and evaluation with its terms kept apart."""

from __future__ import annotations

from collections.abc import Sequence

from satedge_workflow_scheduler.domain.errors import InstanceError
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.requests import Priority, Request
from satedge_workflow_scheduler.domain.solution import ObjectiveValue, RequestOutcome
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode


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


def response_epsilon(requests: Sequence[Request]) -> float:
    """Return ``ε = κ_min / (1 + Σ_r (dl_r − rel_r))``, the weight of the response time.

    It is small enough that the response time never changes the order of two schedules with
    different rejection costs: the rejection costs dominate, and the response time breaks ties.
    The cost ``κ_min`` is the smallest rejection cost, as in the appendix on ε of the document.

    Raises:
        InstanceError: If there are no requests.
    """
    if not requests:
        raise InstanceError("the response weight needs at least one request")
    kappa_min = min(rejection_weights(requests).values())
    slack = sum(r.deadline - r.release for r in requests)
    return kappa_min / (1.0 + slack)


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
        InstanceError: If an outcome refers to a request that is not in the instance.
    """
    requests = {r.id: r for r in instance.requests}
    for outcome in outcomes:
        if outcome.request_id not in requests:
            raise InstanceError(f"outcome for unknown request {outcome.request_id!r}")

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
