"""An interactive page of one solution: the network, the route of each request, the Gantt charts.

The page is a single file with no external resources. This module only prepares its data, as
JSON, in the pixel coordinates of the map; the structure, the style and the interaction live in
the template ``page.html`` next to this module. The geometry (where the nodes are, where each
arrow runs) comes from the same functions the static figures use.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import WindowedAvailability
from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.plotting.data import Span, Timeline, route_of
from satedge_workflow_scheduler.plotting.style import PALETTE
from satedge_workflow_scheduler.plotting.topology import hop_segments, node_positions

_MINUTE = 60.0
_SCALE = 95.0
"""Pixels per unit of the map."""
_MARGIN = 70.0
"""Pixels of empty border around the map, for the labels of the outer nodes."""
_TEMPLATE = Path(__file__).with_name("page.html")

type Point = tuple[float, float]


def page_data(instance: ProblemInstance, data: Timeline) -> dict[str, object]:
    """The data of the page, with times in minutes and the map in pixels.

    Each request gets a colour slot while there are at most as many requests as colours;
    past that every request gets slot ``-1`` and the page shows only the selected one in
    colour, so that two requests never share a colour.
    """
    positions = node_positions(instance)
    to_px = _projection(positions)
    centres = _plane_centres(instance, positions)
    ground = instance.network.ground_station.id
    coms = {plane.com for plane in instance.network.planes}
    coloured = len(data.requests) <= len(PALETTE)
    releases = {r.id: r.release / _MINUTE for r in instance.requests}

    nodes = []
    for node, point in positions.items():
        x, y = to_px(point)
        out_x, out_y = _outward(point, centres.get(node))
        nodes.append(
            {
                "id": node,
                "x": x,
                "y": y,
                "kind": "gs" if node == ground else ("com" if node in coms else "sat"),
                "ox": out_x,
                "oy": out_y,
            }
        )

    arcs = []
    seen: set[frozenset[NodeId]] = set()
    for arc in instance.network.arcs:
        pair = frozenset((arc.source, arc.target))
        if pair in seen:
            continue
        seen.add(pair)
        (x1, y1), (x2, y2) = to_px(positions[arc.source]), to_px(positions[arc.target])
        windows = None
        if isinstance(arc.availability, WindowedAvailability):
            windows = _minutes(
                Span(w.start, w.end) for w in arc.availability.windows.windows
            )
        arcs.append(
            {
                "s": arc.source,
                "t": arc.target,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "windows": windows,
            }
        )

    requests = []
    for index, request_id in enumerate(data.requests):
        plan = route_of(data, request_id)
        segments = hop_segments(instance, plan)
        requests.append(
            {
                "id": request_id,
                "slot": index if coloured else -1,
                "urgent": request_id in data.urgent,
                "release": releases.get(request_id, 0.0),
                "tasks": [
                    {
                        "sat": t.satellite,
                        "k": t.task_index + 1,
                        "start": t.start / _MINUTE,
                        "end": t.end / _MINUTE,
                    }
                    for t in plan.tasks
                ],
                "hops": [
                    {
                        "n": h.number,
                        "s": h.source,
                        "t": h.target,
                        "data": _data_name(h.data_index),
                        "start": h.start / _MINUTE,
                        "end": h.end / _MINUTE,
                        "arrival": h.arrival / _MINUTE,
                        "x1": to_px(segments[h.number][0])[0],
                        "y1": to_px(segments[h.number][0])[1],
                        "x2": to_px(segments[h.number][1])[0],
                        "y2": to_px(segments[h.number][1])[1],
                    }
                    for h in plan.hops
                ],
            }
        )

    pixels = [to_px(point) for point in positions.values()]
    xs = [x for x, _ in pixels]
    ys = [y for _, y in pixels]
    return {
        "name": instance.name or "(senza nome)",
        "horizon": data.horizon / _MINUTE,
        "ground": ground,
        "view": {"width": max(xs) + _MARGIN, "height": max(ys) + _MARGIN},
        "planes": [
            {
                "label": f"Piano {i + 1}",
                "x": to_px(c)[0],
                "y": to_px((c[0], c[1] - 2.25))[1],
            }
            for i, c in enumerate(_plane_centre_list(instance, positions))
        ],
        "nodes": nodes,
        "arcs": arcs,
        "requests": requests,
        "computing": [
            {
                "label": row.satellite,
                "windows": _minutes(row.windows),
                "items": [
                    {
                        "req": t.request_id,
                        "what": f"task {t.task_index + 1}",
                        "start": t.start / _MINUTE,
                        "end": t.end / _MINUTE,
                    }
                    for t in row.tasks
                ],
            }
            for row in data.hpc
        ],
        "links": [
            {
                "label": f"{arc.source} → {arc.target}",
                "windows": None if arc.windows is None else _minutes(arc.windows),
                "items": [
                    {
                        "req": b.request_id,
                        "what": _data_name(b.data_index),
                        "start": b.start / _MINUTE,
                        "end": b.arrival / _MINUTE,
                    }
                    for b in arc.transfers
                ],
            }
            for arc in data.arcs
        ],
    }


def write_page(path: str | Path, instance: ProblemInstance, data: Timeline) -> Path:
    """Write the interactive page of the solution to ``path``. Returns the path."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(page_data(instance, data), ensure_ascii=False)
    # the data sits inside a script element: "</" must not close it early
    payload = payload.replace("</", "<\\/")
    page = _TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)
    file.write_text(page, encoding="utf-8")
    return file


def _projection(positions: Mapping[NodeId, Point]) -> Callable[[Point], Point]:
    """Map coordinates to pixels: y grows downwards, and the map starts at the margin."""
    min_x = min(x for x, _ in positions.values())
    max_y = max(y for _, y in positions.values())

    def to_px(point: Point) -> Point:
        x, y = point
        return round((x - min_x) * _SCALE + _MARGIN, 1), round(
            (max_y - y) * _SCALE + _MARGIN, 1
        )

    return to_px


def _plane_centre_list(
    instance: ProblemInstance, positions: Mapping[NodeId, Point]
) -> list[Point]:
    centres = []
    for plane in instance.network.planes:
        points = [positions[s.id] for s in plane.satellites]
        centres.append(
            (
                sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points),
            )
        )
    return centres


def _plane_centres(
    instance: ProblemInstance, positions: Mapping[NodeId, Point]
) -> dict[NodeId, Point]:
    by_node: dict[NodeId, Point] = {}
    for plane, centre in zip(
        instance.network.planes, _plane_centre_list(instance, positions), strict=True
    ):
        for satellite in plane.satellites:
            by_node[satellite.id] = centre
    return by_node


def _outward(point: Point, centre: Point | None) -> Point:
    """The unit direction, in pixels, from the centre of the ring to the node: where its label goes."""
    if centre is None:
        return 0.0, -1.0  # the ground station: label above it
    dx, dy = point[0] - centre[0], point[1] - centre[1]
    length = math.hypot(dx, dy) or 1.0
    return round(dx / length, 3), round(-dy / length, 3)


def _minutes(spans: Iterable[Span]) -> list[list[float]]:
    return [[s.start / _MINUTE, s.end / _MINUTE] for s in spans]


def _data_name(index: int) -> str:
    return "dato grezzo" if index == 0 else f"output del task {index}"
