"""Ledger of the reservations made by the priority-EDF / EFT heuristic: HPCs, arcs and PLMMs.

The ledger keeps two layers. The *committed* layer holds the reservations of the requests
already planned. The *tentative* layer holds the reservations of the request being planned
inside a transaction: it is visible to readers of the ledger, and it is discarded unless the
transaction is committed.

Online state (the instant from which each satellite is available) will be preloaded
into the committed layer here; for now every node is available from time 0.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass

from satedge_workflow_scheduler.common.errors import SolverError
from satedge_workflow_scheduler.common.intervals import TimeWindow, WindowSet
from satedge_workflow_scheduler.domain.network import Network
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.scheduling.plmm_occupancy import PlmmOccupancyInterval


@dataclass(frozen=True, slots=True)
class HpcKey:
    """Key of the HPC of a satellite."""

    satellite: NodeId


@dataclass(frozen=True, slots=True)
class ArcKey:
    """Key of an arc of the network."""

    source: NodeId
    target: NodeId


type ResourceKey = HpcKey | ArcKey


class ResourceLedger:
    """Committed reservations of HPCs, arcs and PLMMs, with transactional tentative updates."""

    def __init__(self, network: Network) -> None:
        self._network = network
        self._timelines: dict[ResourceKey, WindowSet] = {}
        self._plmm: dict[NodeId, tuple[PlmmOccupancyInterval, ...]] = {}
        self._pending_intervals: dict[ResourceKey, list[TimeWindow]] = defaultdict(list)
        self._pending_plmm: dict[NodeId, list[PlmmOccupancyInterval]] = defaultdict(
            list
        )
        self._in_transaction = False

    def timeline(self, key: ResourceKey) -> WindowSet:
        """Busy timeline of a resource, with committed and tentative reservations."""
        self._check_key(key)
        committed = self._timelines.get(key, WindowSet())
        pending = self._pending_intervals.get(key, [])
        return WindowSet.merged((*committed.windows, *pending))

    def plmm(self, node: NodeId) -> tuple[PlmmOccupancyInterval, ...]:
        """PLMM occupancies of a satellite, with committed and tentative ones."""
        committed = self._plmm.get(node, ())
        return (*committed, *self._pending_plmm.get(node, ()))

    def available_from(self, node: NodeId) -> float:
        """Instant from which ``node`` can be used. Always 0.0 for now."""
        return 0.0

    @contextmanager
    def transaction(self) -> Generator[Transaction]:
        """Open a transaction. Its reservations are kept only if it is committed.

        Raises:
            RuntimeError: If another transaction is already open.
        """
        if self._in_transaction:
            raise RuntimeError("nested transactions are not supported")
        self._in_transaction = True
        tx = Transaction(self)
        try:
            yield tx
        finally:
            self._pending_intervals.clear()
            self._pending_plmm.clear()
            self._in_transaction = False

    def _check_key(self, key: ResourceKey) -> None:
        if isinstance(key, HpcKey):
            if not self._network.is_satellite(key.satellite):
                raise SolverError(f"HPC key for unknown satellite {key.satellite!r}")
        elif (key.source, key.target) not in self._network.arc_index:
            raise SolverError(
                f"arc key {key.source}->{key.target} is not in the network"
            )

    def _commit(self) -> None:
        for key, intervals in self._pending_intervals.items():
            timeline = self._timelines.get(key, WindowSet())
            for interval in intervals:
                timeline = timeline.union_with(interval)
            self._timelines[key] = timeline
        for node, items in self._pending_plmm.items():
            self._plmm[node] = (*self._plmm.get(node, ()), *items)
        self._pending_intervals.clear()
        self._pending_plmm.clear()


class Transaction:
    """Tentative reservations of one planning attempt. Created by :meth:`ResourceLedger.transaction`."""

    def __init__(self, ledger: ResourceLedger) -> None:
        self._ledger = ledger
        self._committed = False

    def reserve(self, key: ResourceKey, interval: TimeWindow) -> None:
        """Add a tentative busy interval on a resource.

        Raises:
            RuntimeError: If the transaction has already been committed.
        """
        self._check_open()
        self._ledger._check_key(key)
        self._ledger._pending_intervals[key].append(interval)

    def add_occupancies(
        self, node: NodeId, items: Iterable[PlmmOccupancyInterval]
    ) -> None:
        """Add tentative PLMM occupancies on a satellite.

        Raises:
            RuntimeError: If the transaction has already been committed.
        """
        self._check_open()
        self._ledger._pending_plmm[node].extend(items)

    def commit(self) -> None:
        """Make the tentative reservations permanent. The transaction is then closed.

        Raises:
            RuntimeError: If the transaction has already been committed.
        """
        self._check_open()
        self._ledger._commit()
        self._committed = True

    def _check_open(self) -> None:
        if self._committed:
            raise RuntimeError("the transaction has already been committed")
