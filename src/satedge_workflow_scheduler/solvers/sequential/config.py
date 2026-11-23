"""Configuration of one run of the sequential heuristic: which members of the family to use.

:class:`SequentialConfig` is immutable pure data, like the MILP configuration. The defaults give
the priority-EDF / EFT heuristic with the safe-interval router, the member of the family that the
document describes. It knows nothing about instances or solvers: the factory (:mod:`.factory`)
turns its choices into objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from satedge_workflow_scheduler.common.intervals import TIME_TOLERANCE
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.sequential.errors import SequentialConfigError


class Ordering(Enum):
    """How the new requests are sorted before they are planned one at a time."""

    PRIORITY_EDF = "priority_edf"
    """Urgent requests first, then by deadline, release and instance order (EDF)."""


class Placement(Enum):
    """How a task is placed among the candidates of its request."""

    EFT = "eft"
    """Earliest finish time: the candidate that ends first."""


class Routing(Enum):
    """How the data items are routed around the reservations."""

    SAFE_INTERVAL = "safe_interval"
    """Safe-interval routing (SIPP): the data may wait where the PLMM has room."""


@dataclass(frozen=True)
class SequentialConfig:
    """Everything that defines one run of the sequential heuristic.

    Attributes:
        deadline: Hard deadlines (requests may be rejected) or soft ones (all are served).
        ordering: The order in which the new requests are planned.
        placement: How each task is placed among its candidates.
        routing: How the data items are routed.
        max_backtracks: How many times the placement may go back to an earlier task to try its
            next candidate, for a request that does not fit. ``0`` is the greedy placement.
        verify: Whether the calendar is checked against the model before it is returned.
        eps: Tolerance for comparisons between instants.

    Raises:
        SequentialConfigError: If a value is out of range.
    """

    deadline: DeadlineMode = DeadlineMode.HARD
    ordering: Ordering = Ordering.PRIORITY_EDF
    placement: Placement = Placement.EFT
    routing: Routing = Routing.SAFE_INTERVAL
    max_backtracks: int = 100
    verify: bool = True
    eps: float = TIME_TOLERANCE

    def __post_init__(self) -> None:
        if self.max_backtracks < 0:
            raise SequentialConfigError(
                f"max_backtracks must be >= 0, got {self.max_backtracks}"
            )
        if self.eps <= 0:
            raise SequentialConfigError(f"eps must be > 0, got {self.eps}")
