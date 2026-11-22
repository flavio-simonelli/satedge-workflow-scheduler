"""Deadline modes and the time limit that each mode imposes on the delivery of a request."""

from __future__ import annotations

from enum import StrEnum

from satedge_workflow_scheduler.domain.requests import Request


class DeadlineMode(StrEnum):
    """How the deadlines of the requests are enforced.

    Attributes:
        HARD: The delivery must happen by the request deadline; requests that cannot meet
            it are rejected.
        SOFT: Every request is served and must be delivered by the horizon; the lateness
            with respect to the deadline is penalised in the objective.
    """

    HARD = "hard"
    SOFT = "soft"


def delivery_limit(request: Request, mode: DeadlineMode, horizon: float) -> float:
    """Return the time limit of the delivery of ``request``.

    Args:
        request: The request.
        mode: The deadline mode.
        horizon: The delivery horizon of the instance, in seconds.

    Returns:
        The deadline of the request in ``HARD`` mode, the horizon in ``SOFT`` mode.
    """
    if mode is DeadlineMode.HARD:
        return request.deadline
    return horizon


def may_reject(mode: DeadlineMode) -> bool:
    """Whether a request may be rejected under ``mode``.

    With hard deadlines a request that cannot meet its deadline is rejected; with soft
    deadlines every request is served, so none is rejected.
    """
    return mode is DeadlineMode.HARD
