"""Deadline policies: the only part where the hard and the soft modes differ.

A policy tells the solver which time limit a request must meet, whether the raw data of
pending requests must be booked in the PLMM, which instants to retry at after a PLMM failure,
and what to record when a request cannot be planned.
"""

from __future__ import annotations

from typing import Protocol

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.requests import Request
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.priority_edf_eft.ledger import ResourceLedger
from satedge_workflow_scheduler.solvers.priority_edf_eft.planner import PlanFailure


class DeadlinePolicy(Protocol):
    """The rules of one deadline mode."""

    mode: DeadlineMode

    def limit(self, request: Request, instance: ProblemInstance) -> float:
        """Latest instant at which the output of ``request`` may reach the ground."""
        ...

    def reserves_pending_raw(self) -> bool:
        """Whether the raw data of pending requests is booked in their origin PLMM."""
        ...

    def retry_instants(
        self, request: Request, ledger: ResourceLedger, instance: ProblemInstance
    ) -> list[float]:
        """Later instants at which to retry planning ``request`` after a PLMM failure."""
        ...

    def on_failure(self, request: Request, failure: PlanFailure) -> None:
        """Record that ``request`` could not be planned."""
        ...


class HardDeadlinePolicy:
    """Hard deadlines: the limit is the deadline, and an unmet request is rejected."""

    mode = DeadlineMode.HARD

    def limit(self, request: Request, instance: ProblemInstance) -> float:
        return request.deadline

    def reserves_pending_raw(self) -> bool:
        return False

    def retry_instants(
        self, request: Request, ledger: ResourceLedger, instance: ProblemInstance
    ) -> list[float]:
        return []

    def on_failure(self, request: Request, failure: PlanFailure) -> None:
        """A rejected request needs no bookkeeping beyond its outcome."""


class SoftDeadlinePolicy:
    """Soft deadlines: the limit is the horizon, and every request must be planned.

    Args:
        max_retries: Maximum number of retry instants tried after a PLMM failure.
    """

    mode = DeadlineMode.SOFT

    def __init__(self, max_retries: int = 100) -> None:
        self._max_retries = max_retries
        self.reasons: dict[str, str] = {}

    def limit(self, request: Request, instance: ProblemInstance) -> float:
        return instance.horizon

    def reserves_pending_raw(self) -> bool:
        return True

    def retry_instants(
        self, request: Request, ledger: ResourceLedger, instance: ProblemInstance
    ) -> list[float]:
        """Instants after the release and before the horizon at which something frees up.

        Candidates are the ends of the PLMM occupancies already committed on any satellite
        and the starts of the HPC windows of all satellites. They are sorted, without
        duplicates, and limited to ``max_retries``.
        """
        horizon = instance.horizon
        candidates: set[float] = set()
        for satellite in instance.network.satellites:
            candidates.update(window.start for window in satellite.hpc_windows)
            candidates.update(item.end for item in ledger.plmm(satellite.id))
        later = sorted(t for t in candidates if request.release < t < horizon)
        return later[: self._max_retries]

    def on_failure(self, request: Request, failure: PlanFailure) -> None:
        self.reasons[request.id] = failure.reason
