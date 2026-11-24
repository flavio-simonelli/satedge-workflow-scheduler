"""One function per figure: each builds a matplotlib figure from the data of a solution."""

from __future__ import annotations

from collections.abc import Mapping

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.plotting.data import RoutePlan, Timeline
from satedge_workflow_scheduler.plotting.hpc import draw_hpc
from satedge_workflow_scheduler.plotting.plmm import draw_memory
from satedge_workflow_scheduler.plotting.style import LABELS, seconds_per
from satedge_workflow_scheduler.plotting.topology import (
    draw_network,
    draw_route,
    route_legend,
    topology_legend,
)
from satedge_workflow_scheduler.plotting.transfers import draw_transfers

Window = tuple[float, float] | None
"""A part of the time to show, in seconds; ``None`` shows from the first to the last event."""

_ROW = 0.6
"""Height in inches of one row of a Gantt chart."""


def hpc_figure(
    data: Timeline, colors: Mapping[str, str], unit: str = "min", window: Window = None
) -> Figure:
    """The computing Gantt: one row per satellite that computes."""
    fig = Figure(figsize=(12, _ROW * len(data.hpc) + 2.2), layout="constrained")
    ax = fig.add_subplot()
    draw_hpc(ax, data, colors, unit)
    _limit(ax, _span(data, window), unit)
    _legend(fig, colors, urgent=True)
    return fig


def transfers_figure(
    data: Timeline, colors: Mapping[str, str], unit: str = "min", window: Window = None
) -> Figure:
    """The transmissions Gantt: one row per arc that carries data."""
    fig = Figure(
        figsize=(12, _ROW * max(len(data.arcs), 1) + 2.2), layout="constrained"
    )
    ax = fig.add_subplot()
    draw_transfers(ax, data, colors, unit)
    _limit(ax, _span(data, window), unit)
    _legend(fig, colors, urgent=False)
    return fig


def memory_figure(data: Timeline, unit: str = "min", window: Window = None) -> Figure:
    """The PLMM occupancy: one step curve per satellite that holds data, on one time axis."""
    fig = Figure(figsize=(12, 6.5), layout="constrained")
    ax = fig.add_subplot()
    draw_memory(ax, data.memory, unit)
    _limit(ax, _span(data, window), unit)
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncols=3, fontsize=12, frameon=False
    )
    return fig


def network_figure(instance: ProblemInstance, at: float | None = None) -> Figure:
    """The constellation map, with the windowed links shown as they are at ``at`` seconds."""
    fig = Figure(figsize=(11, 8), layout="constrained")
    ax = fig.add_subplot()
    draw_network(ax, instance, at=at)
    ax.set_title(
        "Rete della costellazione"
        if at is None
        else f"Rete all'istante {at / 60:.1f} min",
        fontsize=15,
    )
    handles = topology_legend(at)
    fig.legend(
        handles=handles, loc="outside lower center", ncols=3, fontsize=11, frameon=False
    )
    return fig


def route_figure(
    instance: ProblemInstance,
    plan: RoutePlan,
    color: str,
    unit: str = "min",
) -> Figure:
    """The path of one request: its tasks, and the transmissions of each data item, in order."""
    fig = Figure(figsize=(11, 11), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[4, 1.6])
    ax = fig.add_subplot(grid[0])
    draw_route(ax, instance, plan, color, unit)
    ax.set_title(f"Percorso della richiesta {plan.request_id}", fontsize=15)
    text = fig.add_subplot(grid[1])
    text.axis("off")
    text.text(
        0.0,
        1.0,
        _route_text(plan, unit),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    handles = route_legend(color, plan.request_id)
    fig.legend(
        handles=handles, loc="outside lower center", ncols=3, fontsize=11, frameon=False
    )
    return fig


def _route_text(plan: RoutePlan, unit: str) -> str:
    scale = seconds_per(unit)
    lines = [f"Esecuzione (tempi in {unit}):"]
    for task in plan.tasks:
        lines.append(
            f"  task {task.task_index + 1}: {task.satellite:<7} "
            f"da {task.start / scale:7.1f}  a {task.end / scale:7.1f}"
        )
    lines.append("")
    lines.append(f"Trasferimenti in ordine (tempi in {unit}):")
    for hop in plan.hops:
        lines.append(
            f"  {hop.number:>2}. {_data_name(hop.data_index):<22} "
            f"{hop.source} → {hop.target:<7} da {hop.start / scale:6.1f} a {hop.end / scale:6.1f}, "
            f"arrivo {hop.arrival / scale:6.1f}"
        )
    return "\n".join(lines)


def _data_name(index: int) -> str:
    """The name of a data item: the raw input, or the output of a task, numbered from 1."""
    return "dato grezzo" if index == 0 else f"output del task {index}"


def _span(data: Timeline, window: Window) -> tuple[float, float]:
    """The part of the time to show: ``window``, or from the first to the last event."""
    if window is not None:
        return window
    starts = [t.start for h in data.hpc for t in h.tasks]
    ends = [t.end for h in data.hpc for t in h.tasks]
    for arc in data.arcs:
        starts += [b.start for b in arc.transfers]
        ends += [b.arrival for b in arc.transfers]
    for curve in data.memory:
        starts += [s.start for s in curve.segments]
        ends += [s.end for s in curve.segments]
    if not starts:
        return 0.0, data.horizon
    margin = 0.02 * (max(ends) - min(starts))
    return max(0.0, min(starts) - margin), min(data.horizon, max(ends) + margin)


def _limit(ax: Axes, span: tuple[float, float], unit: str) -> None:
    scale = seconds_per(unit)
    ax.set_xlim(span[0] / scale, span[1] / scale)


def _legend(fig: Figure, colors: Mapping[str, str], *, urgent: bool) -> None:
    """One row of legend under the figure, so no space is lost above or beside the charts."""
    handles: list[Patch] = [Patch(facecolor=c, label=rid) for rid, c in colors.items()]
    if urgent:
        handles.append(
            Patch(
                facecolor="white", edgecolor="black", hatch="//", label=LABELS["urgent"]
            )
        )
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncols=min(len(handles), 9),
        fontsize=12,
        frameon=False,
    )
