"""The problem instance: network, requests and the parameters that connect them."""

from __future__ import annotations

import math
from dataclasses import dataclass

from satedge_workflow_scheduler.domain.errors import InstanceError
from satedge_workflow_scheduler.domain.network import Network
from satedge_workflow_scheduler.domain.requests import Request


@dataclass(frozen=True)
class ProblemInstance:
    """A complete scheduling instance.

    Attributes:
        network: Orbital planes, ground station and arcs with their availability.
        requests: The requests. Their order defines the FIFO tie-break between requests of
            the same priority class.
        horizon: Delivery horizon ``H`` in seconds: the latest instant an output can reach
            the ground station. It is the limit of the soft deadline and it is independent of
            the request deadlines.
        name: Optional name of the instance.
    """

    network: Network
    requests: tuple[Request, ...]
    horizon: float
    name: str = ""

    def __post_init__(self) -> None:
        if not (math.isfinite(self.horizon) and self.horizon > 0):
            raise InstanceError(f"horizon must be finite and > 0, got {self.horizon}")

        seen: set[str] = set()
        for request in self.requests:
            if request.id in seen:
                raise InstanceError(f"duplicate request id {request.id!r}")
            seen.add(request.id)

            if not self.network.is_satellite(request.origin):
                raise InstanceError(
                    f"request {request.id}: origin {request.origin!r} is not a satellite"
                )

    def request(self, request_id: str) -> Request:
        """Return the request with identifier ``request_id``."""
        for request in self.requests:
            if request.id == request_id:
                return request
        raise InstanceError(f"unknown request {request_id!r}")
