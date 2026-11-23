"""Priority and EDF ordering: urgent requests first, then by deadline (EDF)."""

from __future__ import annotations

from collections.abc import Sequence

from satedge_workflow_scheduler.domain.requests import Priority, Request


class PriorityEdfOrdering:
    """Urgent requests first; within a class by deadline, then release, then instance order.

    The last criterion is the position of the request in the input sequence, which makes the
    order deterministic and FIFO among otherwise equal requests.
    """

    def order(self, requests: Sequence[Request]) -> list[Request]:
        indexed = sorted(
            enumerate(requests),
            key=lambda item: (
                0 if item[1].priority is Priority.URGENT else 1,
                item[1].deadline,
                item[1].release,
                item[0],
            ),
        )
        return [request for _, request in indexed]
