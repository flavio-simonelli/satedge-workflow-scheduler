"""The problem instance: network, requests and the parameters that connect them."""

from __future__ import annotations

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
        name: Optional name of the instance.
    """

    network: Network
    requests: tuple[Request, ...]
    name: str = ""

    def __post_init__(self) -> None:
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

    @property
    def horizon(self) -> float:
        """Latest deadline of the instance, in seconds (0 without requests)."""
        return max((request.deadline for request in self.requests), default=0.0)
