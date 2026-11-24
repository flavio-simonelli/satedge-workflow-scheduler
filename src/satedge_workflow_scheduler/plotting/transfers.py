"""Gantt of the transmissions: one row per arc that carries data."""

from __future__ import annotations

from collections.abc import Mapping

from matplotlib.axes import Axes

from satedge_workflow_scheduler.plotting.data import Timeline
from satedge_workflow_scheduler.plotting.style import (
    BAND,
    LABEL_SIZE,
    LABELS,
    TICK_SIZE,
    TITLE_SIZE,
    seconds_per,
    time_label,
)


def draw_transfers(
    ax: Axes, data: Timeline, colors: Mapping[str, str], unit: str
) -> None:
    """Draw the visibility windows in grey, each transmission, and its latency tail.

    An arc that is always available has no band behind it.
    """
    scale = seconds_per(unit)
    for row, arc in enumerate(data.arcs):
        for window in arc.windows or ():
            ax.barh(
                row,
                (window.end - window.start) / scale,
                left=window.start / scale,
                height=0.8,
                color=BAND,
                linewidth=0,
                zorder=1,
            )
        for bar in arc.transfers:
            color = colors[bar.request_id]
            ax.barh(
                row,
                (bar.end - bar.start) / scale,
                left=bar.start / scale,
                height=0.4,
                color=color,
                zorder=2,
            )
            ax.plot(
                [bar.end / scale, bar.arrival / scale],
                [row, row],
                color=color,
                linewidth=2.0,
                zorder=2,
            )
    ax.set_yticks(
        range(len(data.arcs)), [f"{arc.source} → {arc.target}" for arc in data.arcs]
    )
    ax.invert_yaxis()
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlabel(time_label(unit), fontsize=LABEL_SIZE)
    ax.set_title(
        f"{LABELS['transmission']} (visibilità in grigio)", fontsize=TITLE_SIZE
    )
