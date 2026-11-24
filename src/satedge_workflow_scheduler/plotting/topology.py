"""The network as a map, and the path of one request over it."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import Arc, WindowedAvailability
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.plotting.data import Hop, RoutePlan
from satedge_workflow_scheduler.plotting.style import seconds_per

_PLANE_SPACING = 5.0
_RING_RADIUS = 1.7
_GROUND_Y = 4.6
_FAINT = "#c8c8c8"
_LANE = 0.2
"""Distance in the map between the two directions of one arc, and between repeated transmissions."""
ROUTE_BLUE = "#1f3b73"
"""Colour of the routes when no request is singled out."""


@dataclass(frozen=True)
class _LinkStyle:
    color: str
    linewidth: float
    linestyle: str


def draw_network(
    ax: Axes, instance: ProblemInstance, *, at: float | None = None
) -> None:
    """Draw the constellation: the planes as rings, the COMs, and the ground station.

    With ``at`` a windowed link is solid when open at that instant and dotted when closed;
    without it, a windowed link is dashed.
    """
    positions = _positions(instance)
    drawn: set[frozenset[NodeId]] = set()
    for arc in instance.network.arcs:
        pair = frozenset((arc.source, arc.target))
        if pair not in drawn:
            drawn.add(pair)
            _segment(ax, positions, arc.source, arc.target, _link_style(arc, at))
    _nodes(ax, instance, positions)
    for index, _plane in enumerate(instance.network.planes):
        ax.text(
            index * _PLANE_SPACING,
            -2.9,
            f"Piano {index + 1}",
            ha="center",
            fontsize=12,
            weight="bold",
        )
    ax.set_aspect("equal")
    ax.axis("off")


def draw_route(
    ax: Axes, instance: ProblemInstance, plan: RoutePlan, color: str, unit: str
) -> None:
    """Draw the path of ``plan`` over the network.

    Each transmission is an arrow in the colour of the request, numbered in time order. Two
    transmissions in opposite directions on the same arc run side by side, as two straight
    arrows. The satellite that runs a task is named by a label above it, with the times.
    """
    positions = _positions(instance)
    draw_network(ax, instance, at=None)
    scale = seconds_per(unit)
    for hop in plan.hops:
        start, end = _lane(plan, hop, positions)
        _arrow(ax, start, end, color, instantaneous=hop.end - hop.start < 1.0)
    for hop in plan.hops:
        start, end = _lane(plan, hop, positions)
        _number(ax, (start, end), hop.number, color)
    for task in plan.tasks:
        x, y = positions[task.satellite]
        label = f"task {task.task_index + 1}  {task.start / scale:.1f}–{task.end / scale:.1f} {unit}"
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 24),
            ha="center",
            va="bottom",
            fontsize=11,
            color=color,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": color,
            },
            zorder=6,
        )
        ax.scatter(
            [x],
            [y],
            s=520,
            marker="o",
            color=color,
            edgecolor="black",
            linewidth=2.2,
            zorder=7,
        )


def route_legend(color: str, request_id: str) -> list[Line2D]:
    """The legend of a route: the arrows, and the marks of the nodes."""
    return [
        Line2D([0], [0], color=color, linewidth=3, label=f"percorso di {request_id}"),
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markeredgecolor="black",
            linestyle="",
            markersize=12,
            label="satellite che esegue un task",
        ),
        Line2D([0], [0], color=_FAINT, linewidth=1.2, label="altri collegamenti"),
    ]


def topology_legend(at: float | None) -> list[Line2D]:
    """The legend of the network map; the window entries only when an instant is given."""
    handles = [
        Line2D(
            [0],
            [0],
            color="#8c8c8c",
            linewidth=1.2,
            label="collegamento sempre disponibile",
        )
    ]
    if at is None:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#8c8c8c",
                linewidth=1.2,
                linestyle="--",
                label="collegamento a finestre",
            )
        )
    else:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=1.4,
                label="finestra aperta all'istante",
            )
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=_FAINT,
                linewidth=1.2,
                linestyle=":",
                label="finestra chiusa all'istante",
            )
        )
    return handles


def _positions(instance: ProblemInstance) -> dict[NodeId, tuple[float, float]]:
    """Each plane is a ring with its COM on top; the ground station sits above the planes."""
    positions: dict[NodeId, tuple[float, float]] = {}
    planes = instance.network.planes
    for index, plane in enumerate(planes):
        cx = index * _PLANE_SPACING
        com_at = next(i for i, s in enumerate(plane.satellites) if s.id == plane.com)
        count = len(plane.satellites)
        for position, satellite in enumerate(plane.satellites):
            angle = math.pi / 2 + 2 * math.pi * (position - com_at) / count
            positions[satellite.id] = (
                cx + _RING_RADIUS * math.cos(angle),
                _RING_RADIUS * math.sin(angle),
            )
    last = (len(planes) - 1) * _PLANE_SPACING
    positions[instance.network.ground_station.id] = (last / 2, _GROUND_Y)
    return positions


def _nodes(
    ax: Axes, instance: ProblemInstance, positions: Mapping[NodeId, tuple[float, float]]
) -> None:
    coms = {plane.com for plane in instance.network.planes}
    ground = instance.network.ground_station.id
    for node, (x, y) in positions.items():
        if node == ground:
            ax.scatter(
                [x], [y], s=260, marker="*", color="white", edgecolor="black", zorder=4
            )
        elif node in coms:
            ax.scatter(
                [x],
                [y],
                s=120,
                marker="s",
                color="white",
                edgecolor="black",
                linewidth=1.5,
                zorder=4,
            )
        else:
            ax.scatter(
                [x], [y], s=60, marker="o", color="white", edgecolor="black", zorder=4
            )
        side = 1 if x >= 0 else -1
        ax.annotate(
            node,
            (x, y),
            textcoords="offset points",
            xytext=(10 * side, 0),
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=11,
            weight="bold" if node in coms or node == ground else "normal",
            zorder=5,
        )


def _segment(
    ax: Axes,
    positions: Mapping[NodeId, tuple[float, float]],
    source: NodeId,
    target: NodeId,
    style: _LinkStyle,
) -> None:
    (x1, y1), (x2, y2) = positions[source], positions[target]
    ax.plot(
        [x1, x2],
        [y1, y2],
        color=style.color,
        linewidth=style.linewidth,
        linestyle=style.linestyle,
        zorder=1,
    )


def _lane(
    plan: RoutePlan, hop: Hop, positions: Mapping[NodeId, tuple[float, float]]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two ends of the arrow of ``hop``, moved sideways so that it does not sit on its twin.

    A transmission alone on its arc stays on the arc. When the arc is used in both directions,
    each direction runs on its own lane, on opposite sides; repeated transmissions in one
    direction are spread across further lanes.
    """
    (x1, y1), (x2, y2) = positions[hop.source], positions[hop.target]
    reverse = any((h.source, h.target) == (hop.target, hop.source) for h in plan.hops)
    if not reverse:
        return (x1, y1), (x2, y2)
    same = [h for h in plan.hops if (h.source, h.target) == (hop.source, hop.target)]
    side = 1.0 if hop.source < hop.target else -1.0
    shift = side * (_LANE + _LANE * same.index(hop))
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    nx, ny = -(y2 - y1) / length * shift, (x2 - x1) / length * shift
    return (x1 + nx, y1 + ny), (x2 + nx, y2 + ny)


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    instantaneous: bool,
) -> None:
    """A straight arrow; a transmission that lasts under a second is dotted."""
    annotation = ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 3,
            "mutation_scale": 22,
            "shrinkA": 6,
            "shrinkB": 6,
        },
        zorder=3,
    )
    if instantaneous and annotation.arrow_patch is not None:
        annotation.arrow_patch.set_linestyle((0, (1, 2)))


def _number(
    ax: Axes,
    ends: tuple[tuple[float, float], tuple[float, float]],
    number: int,
    color: str,
) -> None:
    """The number of a transmission, at the middle of its arrow."""
    (x1, y1), (x2, y2) = ends
    ax.text(
        x1 + 0.4 * (x2 - x1),
        y1 + 0.4 * (y2 - y1),
        str(number),
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        weight="bold",
        bbox={"boxstyle": "circle,pad=0.25", "facecolor": color, "edgecolor": "black"},
        zorder=6,
    )


def _link_style(arc: Arc, at: float | None) -> _LinkStyle:
    if not isinstance(arc.availability, WindowedAvailability):
        return _LinkStyle("#8c8c8c", 1.2, "-")
    if at is None:
        return _LinkStyle("#8c8c8c", 1.2, "--")
    open_now = any(w.start <= at < w.end for w in arc.availability.windows.windows)
    return _LinkStyle("#333333", 1.4, "-") if open_now else _LinkStyle(_FAINT, 1.2, ":")
