"""Resources already reserved by the requests planned so far: HPCs, arcs and PLMMs.

A sequential heuristic plans one request at a time around what the previous requests reserved.
:class:`Reservations` is immutable: every reservation returns a new object, so a tentative plan
is just another object, kept if the request is planned and dropped if not.

- An HPC or an arc is an exclusive resource: it holds a :class:`BusyTimeline`, and a new task or
  transmission must avoid it.
- A PLMM is a cumulative resource: it holds the occupancy intervals of the data, and a new
  stay fits if the load stays within the capacity of the satellite. The room is given by the
  free intervals of the PLMM (:func:`free_intervals`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from satedge_workflow_scheduler.common.intervals import BusyTimeline, TimeWindow
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.nodes import NodeId, Satellite
from satedge_workflow_scheduler.domain.solution import Schedule, TaskExecution, Transfer
from satedge_workflow_scheduler.scheduling.plmm_occupancy import (
    FreeInterval,
    PlmmOccupancyInterval,
    free_intervals,
    plmm_intervals,
)

_FREE = BusyTimeline()

type ArcId = tuple[NodeId, NodeId]


class Reservations:
    """Immutable reservations of the HPCs, the arcs and the PLMMs of the network."""

    __slots__ = ("_arcs", "_hpc", "_plmm")

    def __init__(
        self,
        hpc: Mapping[NodeId, BusyTimeline] | None = None,
        arcs: Mapping[ArcId, BusyTimeline] | None = None,
        plmm: Mapping[NodeId, tuple[PlmmOccupancyInterval, ...]] | None = None,
    ) -> None:
        self._hpc: Mapping[NodeId, BusyTimeline] = dict(hpc or {})
        self._arcs: Mapping[ArcId, BusyTimeline] = dict(arcs or {})
        self._plmm: Mapping[NodeId, tuple[PlmmOccupancyInterval, ...]] = dict(
            plmm or {}
        )

    # --- reading ---------------------------------------------------------------------

    def hpc_busy(self, satellite: NodeId) -> BusyTimeline:
        """Intervals in which the HPC of ``satellite`` already runs a task."""
        return self._hpc.get(satellite, _FREE)

    def arc_busy(self, source: NodeId, target: NodeId) -> BusyTimeline:
        """Intervals in which the arc ``source -> target`` already transmits."""
        return self._arcs.get((source, target), _FREE)

    def plmm_occupancies(self, satellite: NodeId) -> tuple[PlmmOccupancyInterval, ...]:
        """Occupancies of the PLMM of ``satellite``, half-open intervals."""
        return self._plmm.get(satellite, ())

    def plmm_free_intervals(
        self, satellite: Satellite, size: float, eps: float
    ) -> tuple[FreeInterval, ...]:
        """The intervals in which ``size`` Gb more fit in the PLMM of ``satellite``."""
        return free_intervals(
            self.plmm_occupancies(satellite.id), satellite.plmm_capacity, size, eps
        )

    def plmm_has_room(
        self, satellite: Satellite, size: float, start: float, end: float, eps: float
    ) -> bool:
        """Whether ``size`` Gb more fit in the PLMM of ``satellite`` during ``[start, end)``."""
        return any(
            interval.covers(start, end, eps)
            for interval in self.plmm_free_intervals(satellite, size, eps)
        )

    # --- reserving -------------------------------------------------------------------

    def with_execution(self, execution: TaskExecution) -> Reservations:
        """Also reserve the HPC for ``execution``."""
        hpc = dict(self._hpc)
        hpc[execution.satellite] = self.hpc_busy(execution.satellite).with_interval(
            _span(execution.start, execution.end)
        )
        return Reservations(hpc, self._arcs, self._plmm)

    def with_transfer(self, transfer: Transfer) -> Reservations:
        """Also reserve the arc of ``transfer`` while it transmits."""
        arcs = dict(self._arcs)
        key = (transfer.source, transfer.target)
        arcs[key] = self.arc_busy(*key).with_interval(
            _span(transfer.start, transfer.end)
        )
        return Reservations(self._hpc, arcs, self._plmm)

    def with_occupancies(
        self, occupancies: Iterable[tuple[NodeId, PlmmOccupancyInterval]]
    ) -> Reservations:
        """Also reserve the PLMM ``occupancies``, each on its satellite."""
        plmm = dict(self._plmm)
        for satellite, item in occupancies:
            plmm[satellite] = (*plmm.get(satellite, ()), item)
        return Reservations(self._hpc, self._arcs, plmm)

    def with_schedule(
        self, instance: ProblemInstance, schedule: Schedule
    ) -> Reservations:
        """Also reserve everything ``schedule`` uses: HPCs, arcs and PLMMs.

        The PLMM occupancies are the ones the model gives to the schedule
        (:func:`plmm_intervals`). This is how a planned request, or a committed calendar, is
        reserved.
        """
        result = self
        for execution in schedule.executions:
            result = result.with_execution(execution)
        for transfer in schedule.transfers:
            result = result.with_transfer(transfer)
        occupancies = plmm_intervals(instance, schedule.executions, schedule.transfers)
        return result.with_occupancies(
            (satellite, item)
            for satellite, items in occupancies.items()
            for item in items
        )

    def without_plmm(self) -> Reservations:
        """The same HPC and arc reservations, with every PLMM empty."""
        return Reservations(self._hpc, self._arcs)

    def without_occupancies_of(self, request_id: str) -> Reservations:
        """The same reservations without the PLMM occupancies of ``request_id``."""
        plmm = {
            satellite: tuple(i for i in items if i.request_id != request_id)
            for satellite, items in self._plmm.items()
        }
        return Reservations(self._hpc, self._arcs, plmm)


def _span(start: float, end: float) -> TimeWindow:
    return TimeWindow(start, end)
