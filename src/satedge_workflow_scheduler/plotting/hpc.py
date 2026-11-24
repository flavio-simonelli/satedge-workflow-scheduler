"""Gantt of the computing: one row per satellite that computes."""

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


def draw_hpc(ax: Axes, data: Timeline, colors: Mapping[str, str], unit: str) -> None:
    """Draw the computing windows in grey, and each task in the colour of its request.

    Urgent tasks are hatched, so they stand out in a black and white print too.
    """
    scale = seconds_per(unit)
    for row, satellite in enumerate(data.hpc):
        for window in satellite.windows:
            ax.barh(
                row,
                (window.end - window.start) / scale,
                left=window.start / scale,
                height=0.8,
                color=BAND,
                linewidth=0,
                zorder=1,
            )
        for task in satellite.tasks:
            ax.barh(
                row,
                (task.end - task.start) / scale,
                left=task.start / scale,
                height=0.55,
                color=colors[task.request_id],
                edgecolor="black",
                linewidth=0.8,
                hatch="//" if task.urgent else None,
                zorder=2,
            )
    ax.set_yticks(range(len(data.hpc)), [s.satellite for s in data.hpc])
    ax.invert_yaxis()
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlabel(time_label(unit), fontsize=LABEL_SIZE)
    ax.set_title(f"{LABELS['hpc']} (finestre in grigio)", fontsize=TITLE_SIZE)
