"""PLMM occupancy: one step curve per satellite, on the same time axis."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.axes import Axes

from satedge_workflow_scheduler.plotting.data import MemoryCurve
from satedge_workflow_scheduler.plotting.style import (
    LABEL_SIZE,
    LABELS,
    PALETTE,
    TICK_SIZE,
    seconds_per,
    time_label,
)


def draw_memory(ax: Axes, curves: Sequence[MemoryCurve], unit: str) -> None:
    """Draw the load of each satellite as a step curve, with its peak in the legend.

    Between two stretches of data the curve drops to zero, so the gaps are visible.
    """
    scale = seconds_per(unit)
    for index, curve in enumerate(curves):
        xs: list[float] = []
        ys: list[float] = []
        for segment in curve.segments:
            if not xs or segment.start > xs[-1]:
                xs += [segment.start / scale, segment.start / scale]
                ys += [0.0, segment.load]
            else:
                ys[-1] = segment.load
            xs.append(segment.end / scale)
            ys.append(segment.load)
            xs.append(segment.end / scale)
            ys.append(0.0)
        ax.plot(
            xs,
            ys,
            color=PALETTE[index % len(PALETTE)],
            linewidth=1.8,
            label=f"{curve.satellite} ({LABELS['peak']} {curve.peak:.1f} Gb)",
        )
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlabel(time_label(unit), fontsize=LABEL_SIZE)
    ax.set_ylabel("Occupazione PLMM [Gb]", fontsize=LABEL_SIZE)
    ax.set_title(LABELS["memory"], fontsize=LABEL_SIZE + 1)
