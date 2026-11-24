"""Prototype of the figures on the case study, solved with the sequential heuristic.

Usage: python scripts/prototipo_grafici.py [--unit min|s|h] [--out output/prototipo]

It writes the Gantt charts and the PLMM curves for the whole solution, the map of the network,
and one path figure per request.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from satedge_workflow_scheduler import api
from satedge_workflow_scheduler.io import load_instance
from satedge_workflow_scheduler.plotting.data import route_of, timeline
from satedge_workflow_scheduler.plotting.figures import (
    hpc_figure,
    memory_figure,
    network_figure,
    route_figure,
    transfers_figure,
)
from satedge_workflow_scheduler.plotting.save import save
from satedge_workflow_scheduler.plotting.style import request_colors
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig

CASE = Path("examples/casostudio")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", default="min", choices=["s", "min", "h"])
    parser.add_argument("--out", type=Path, default=Path("output/prototipo"))
    args = parser.parse_args()

    instance = load_instance(
        CASE / "instance.yaml", contact_plan=CASE / "contact_plan.yaml"
    )
    result = api.solve(instance, SequentialConfig())
    data = timeline(instance, result.solution)
    colors = request_colors(data.requests)

    figures = {
        "hpc": hpc_figure(data, colors, args.unit),
        "trasferimenti": transfers_figure(data, colors, args.unit),
        "plmm": memory_figure(data, args.unit),
        "rete": network_figure(instance),
    }
    for request_id in data.requests:
        figures[f"percorso-{request_id}"] = route_figure(
            instance, route_of(data, request_id), colors[request_id], args.unit
        )
    for name, figure in figures.items():
        print(save(figure, args.out / f"{name}.png"))


if __name__ == "__main__":
    main()
