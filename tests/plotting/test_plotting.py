"""The plotting data and the figures: the values agree with the solution they come from."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sample import sample_instance

from satedge_workflow_scheduler import api
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.plotting.data import Timeline, route_of, timeline
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig


def _solved() -> tuple[ProblemInstance, api.Result, Timeline]:
    instance = sample_instance()
    result = api.solve(instance, SequentialConfig())
    return instance, result, timeline(instance, result.solution)


def test_every_execution_and_transfer_is_in_the_timeline() -> None:
    _, result, data = _solved()
    tasks = sum(len(h.tasks) for h in data.hpc)
    hops = sum(len(a.transfers) for a in data.arcs)
    assert tasks == len(result.solution.schedule.executions)
    assert hops == len(result.solution.schedule.transfers)


def test_the_arrival_of_a_transfer_is_its_end_plus_the_latency() -> None:
    instance, _, data = _solved()
    latency = {(a.source, a.target): a.latency for a in instance.network.arcs}
    for arc in data.arcs:
        for bar in arc.transfers:
            assert bar.arrival == pytest.approx(
                bar.end + latency[(arc.source, arc.target)]
            )


def test_the_route_of_a_request_is_numbered_in_time_order() -> None:
    _, _, data = _solved()
    plan = route_of(data, data.requests[0])
    assert [h.number for h in plan.hops] == list(range(1, len(plan.hops) + 1))
    starts = [h.start for h in plan.hops]
    assert starts == sorted(starts)


def test_an_unknown_request_is_rejected() -> None:
    _, _, data = _solved()
    with pytest.raises(ValueError, match="unknown request"):
        route_of(data, "missing")


def test_the_figures_are_written_as_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from satedge_workflow_scheduler.plotting.figures import hpc_figure, network_figure
    from satedge_workflow_scheduler.plotting.save import save
    from satedge_workflow_scheduler.plotting.style import request_colors

    instance, _, data = _solved()
    colors = request_colors(data.requests)
    for name, figure in {
        "hpc": hpc_figure(data, colors),
        "rete": network_figure(instance),
    }.items():
        path = save(figure, tmp_path / f"{name}.png")
        assert path.stat().st_size > 1000


def test_the_plotting_modules_do_not_load_matplotlib_on_import() -> None:
    import importlib

    for name in list(sys.modules):
        if name.startswith("matplotlib"):
            pytest.skip("matplotlib already imported by another test")
    importlib.import_module("satedge_workflow_scheduler.plotting.data")
    assert not any(name.startswith("matplotlib") for name in sys.modules)


def test_the_page_holds_every_request_and_its_route(tmp_path: Path) -> None:
    from satedge_workflow_scheduler.plotting.html import page_data, write_page

    instance, result, data = _solved()
    page = page_data(instance, data)
    requests = page["requests"]
    assert isinstance(requests, list)
    assert [r["id"] for r in requests] == list(data.requests)
    hops = sum(len(r["hops"]) for r in requests)
    assert hops == len(result.solution.schedule.transfers)

    file = write_page(tmp_path / "page.html", instance, data)
    text = file.read_text(encoding="utf-8")
    assert "__DATA__" not in text
    # the data is embedded in one script element that nothing in it can close early
    assert text.count("</script>") == 2
