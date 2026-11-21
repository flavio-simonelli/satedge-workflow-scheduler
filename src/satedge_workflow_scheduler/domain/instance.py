"""The problem instance: network, requests and the parameters that connect them."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from satedge_workflow_scheduler.common.errors import InstanceError
from satedge_workflow_scheduler.domain.network import Network
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.domain.solution import Schedule


@dataclass(frozen=True)
class ProblemInstance:
    """A complete scheduling instance, offline or of one planning epoch.

    Attributes:
        network: Orbital planes, ground station and arcs with their availability.
        requests: The requests. Their order defines the FIFO tie-break between requests of
            the same priority class.
        horizon: Scheduling horizon ``H`` in seconds: the latest instant an output can reach
            the ground station. It is the limit of the soft deadline and it is independent of
            the request deadlines.
        plan_release: Plan release ``b^plan_s`` of each satellite, in seconds: the instant at
            which the new plan reaches it, before which no new task or transmission can start
            there. ``math.inf`` means that the plan does not reach the satellite. A satellite
            not listed has release 0.
        committed: Committed calendar, from a previous plan and fixed. The requests that
            appear in it are the committed ones, and each has one execution per task.
        name: Optional name of the instance.
    """

    network: Network
    requests: tuple[Request, ...]
    horizon: float
    plan_release: Mapping[NodeId, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    committed: Schedule = field(default_factory=Schedule)
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

        # read-only copy: the caller's dict can change without affecting the instance
        object.__setattr__(
            self, "plan_release", MappingProxyType(dict(self.plan_release))
        )
        for node, release in self.plan_release.items():
            if not self.network.is_satellite(node):
                raise InstanceError(
                    f"plan release of {node!r}, which is not a satellite"
                )
            if math.isnan(release) or release < 0:
                raise InstanceError(
                    f"plan release of {node!r} must be >= 0 or inf, got {release}"
                )

        self._check_committed()

    def _check_committed(self) -> None:
        requests = {r.id: r for r in self.requests}
        for request_id in self.committed.request_ids:
            request = requests.get(request_id)
            if request is None:
                raise InstanceError(
                    f"committed calendar refers to unknown request {request_id!r}"
                )
            task_indices = sorted(
                e.task_index
                for e in self.committed.executions
                if e.request_id == request_id
            )
            if task_indices != list(range(len(request.tasks))):
                raise InstanceError(
                    f"committed request {request_id}: needs one execution per task, "
                    f"got tasks {task_indices}"
                )

    def release_of(self, satellite: NodeId) -> float:
        """Plan release of ``satellite``: 0 if not listed."""
        return self.plan_release.get(satellite, 0.0)

    def is_committed(self, request_id: str) -> bool:
        """Whether the request appears in the committed calendar."""
        return request_id in self.committed.request_ids

    @property
    def committed_requests(self) -> tuple[Request, ...]:
        """The committed requests, in instance order."""
        committed = self.committed.request_ids
        return tuple(r for r in self.requests if r.id in committed)

    @property
    def new_requests(self) -> tuple[Request, ...]:
        """The requests still to plan, in instance order."""
        committed = self.committed.request_ids
        return tuple(r for r in self.requests if r.id not in committed)

    def request(self, request_id: str) -> Request:
        """Return the request with identifier ``request_id``."""
        for request in self.requests:
            if request.id == request_id:
                return request
        raise InstanceError(f"unknown request {request_id!r}")
