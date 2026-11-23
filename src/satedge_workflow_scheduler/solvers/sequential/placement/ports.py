"""What a scheduler of one request gives back: the plan, or why it failed.

A scheduler places the tasks of one request around the reservations of the requests planned
before it. It is the second choice of the family, after the order (:mod:`..ordering`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.domain.solution import Schedule
from satedge_workflow_scheduler.scheduling.bounds import RequestBounds
from satedge_workflow_scheduler.solvers.sequential.reservations import Reservations


@dataclass(frozen=True)
class RequestPlan:
    """A planned request.

    Attributes:
        request_id: Identifier of the request.
        schedule: Its executions and transfers, delivery to the ground included.
        delivery_time: When its final output reaches the ground station.
    """

    request_id: str
    schedule: Schedule
    delivery_time: float


@dataclass(frozen=True)
class PlanFailure:
    """A request that could not be planned, with the reason."""

    reason: str


class RequestScheduler(Protocol):
    """Places one request around the reservations. It keeps no state between calls."""

    def schedule(
        self,
        instance: ProblemInstance,
        request: Request,
        bounds: RequestBounds,
        reservations: Reservations,
    ) -> RequestPlan | PlanFailure:
        """Plan ``request`` by ``bounds.limit`` around ``reservations``."""
        ...
