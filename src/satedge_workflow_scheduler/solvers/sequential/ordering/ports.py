"""The ordering of the sequential heuristic: the rule that sorts the requests before planning.

An ordering only sorts: it does not know the instance, the reservations or the routes. That is
why the family can swap it without touching the planning.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from satedge_workflow_scheduler.domain.requests import Request


class RequestOrdering(Protocol):
    """A rule that decides the order in which requests are planned."""

    def order(self, requests: Sequence[Request]) -> list[Request]:
        """Return the requests in the order in which they must be planned."""
        ...
