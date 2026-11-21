"""Nodes of the constellation: satellites and the ground station."""

from __future__ import annotations

import math
from dataclasses import dataclass

from satedge_workflow_scheduler.common.errors import InstanceError
from satedge_workflow_scheduler.common.intervals import TimeWindow, WindowSet

type NodeId = str


@dataclass(frozen=True)
class Satellite:
    """A satellite node: it computes (HPC), stores (PLMM) and relays data.

    Attributes:
        id: Unique node identifier.
        plmm_capacity: Net PLMM capacity available to request data, in Gb (> 0).
        hpc_rate: Rate in Gbps of the HPC-router link, the same in both directions: the HPC
            reads its input from the PLMM and writes its output into the PLMM at this rate (> 0).
        hpc_windows: Computing windows (net of container start-up), sorted and disjoint.
    """

    id: NodeId
    plmm_capacity: float
    hpc_rate: float
    hpc_windows: tuple[TimeWindow, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise InstanceError("satellite id must be non-empty")
        if self.plmm_capacity <= 0:
            raise InstanceError(f"satellite {self.id}: PLMM capacity must be > 0")
        if not (math.isfinite(self.hpc_rate) and self.hpc_rate > 0):
            raise InstanceError(f"satellite {self.id}: HPC rate must be finite and > 0")
        try:
            WindowSet(self.hpc_windows)
        except ValueError as error:
            raise InstanceError(f"HPC of satellite {self.id}: {error}") from error

    def hpc_io_time(self, size: float) -> float:
        """Time in seconds to read ``size`` Gb from the PLMM, or to write them into it."""
        return size / self.hpc_rate


@dataclass(frozen=True)
class GroundStation:
    """The ground station: it only receives final outputs (no computing, no relay).

    Attributes:
        id: Unique node identifier.
    """

    id: NodeId

    def __post_init__(self) -> None:
        if not self.id:
            raise InstanceError("ground station id must be non-empty")
