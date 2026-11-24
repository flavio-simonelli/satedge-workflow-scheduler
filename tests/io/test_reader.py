"""Reading an instance file: the domain objects it produces, and the errors it reports."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    WindowedAvailability,
)
from satedge_workflow_scheduler.domain.requests import Priority
from satedge_workflow_scheduler.io import DocumentError, load_instance

TINY = """\
schema_version: "1.0.0"
ground_station: G
hpc_rate_gbps: 1.0
horizon_s: 600
topology:
  kind: explicit
  planes:
    - id: 1
      com: S2
      satellites:
        - {id: S1, plmm_capacity_gb: 10, hpc_windows: [{start_s: 0, end_s: 100}]}
        - {id: S2, plmm_capacity_gb: 10, hpc_windows: [{start_s: 0, end_s: 100}]}
  links:
    - {a: S1, b: S2, capacity_gbps: 1, latency_s: 1}
    - {a: S2, b: G, capacity_gbps: 2, latency_s: 1, windows: [{start_s: 0, end_s: 1000}]}
requests:
  - id: r1
    priority: urgent
    origin: S1
    release_s: 0
    raw_size_gb: 2
    tasks:
      - {execution_s: 10, output_size_gb: 0.5}
plan_release_s: {S2: null}
"""


def _write(tmp_path: Path, text: str, name: str = "tiny.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_the_instance_has_the_nodes_requests_and_horizon_of_the_file(
    tmp_path: Path,
) -> None:
    instance = load_instance(_write(tmp_path, TINY))

    assert [s.id for s in instance.network.satellites] == ["S1", "S2"]
    assert instance.network.ground_station.id == "G"
    assert instance.horizon == 600.0
    (request,) = instance.requests
    assert request.priority is Priority.URGENT
    assert request.deadline == pytest.approx(
        7200.0
    )  # the default delay of urgent requests
    assert request.tasks[0].execution_time == 10.0


def test_a_satellite_pair_gives_two_arcs_and_a_ground_link_gives_one(
    tmp_path: Path,
) -> None:
    network = load_instance(_write(tmp_path, TINY)).network
    pairs = {(a.source, a.target) for a in network.arcs}

    assert pairs == {("S1", "S2"), ("S2", "S1"), ("S2", "G")}
    to_ground = next(a for a in network.arcs if a.target == "G")
    assert isinstance(to_ground.availability, WindowedAvailability)
    assert isinstance(network.arcs[0].availability, AlwaysAvailable)


def test_plan_release_null_is_infinite_and_missing_satellites_get_zero(
    tmp_path: Path,
) -> None:
    instance = load_instance(_write(tmp_path, TINY))

    assert instance.plan_release["S1"] == 0.0
    assert math.isinf(instance.plan_release["S2"])


def test_an_instance_without_a_name_takes_the_name_of_the_file(tmp_path: Path) -> None:
    assert load_instance(_write(tmp_path, TINY, "run_7.yaml")).name == "run_7"


def test_a_name_in_the_file_is_kept(tmp_path: Path) -> None:
    named = TINY.replace("ground_station: G\n", "name: nominal\nground_station: G\n")
    assert load_instance(_write(tmp_path, named, "run_7.yaml")).name == "nominal"


def test_the_error_names_the_field_that_is_wrong(tmp_path: Path) -> None:
    broken = TINY.replace("execution_s: 10", "execution_s: soon")
    broken_file = _write(tmp_path, broken)
    with pytest.raises(DocumentError, match=r"requests\[0\]\.tasks\[0\]\.execution_s"):
        load_instance(broken_file)


def test_an_unknown_key_is_rejected(tmp_path: Path) -> None:
    extra_file = _write(tmp_path, TINY + "colour: red\n")
    with pytest.raises(DocumentError, match="(?i)extra"):
        load_instance(extra_file)


def test_a_newer_major_version_is_rejected(tmp_path: Path) -> None:
    newer_file = _write(tmp_path, TINY.replace('"1.0.0"', '"2.0.0"'))
    with pytest.raises(DocumentError, match="unsupported schema version"):
        load_instance(newer_file)


def test_without_requests_the_horizon_is_required(tmp_path: Path) -> None:
    text = TINY.replace("horizon_s: 600\n", "").split("requests:")[0]
    text_file = _write(tmp_path, text)
    with pytest.raises(DocumentError, match="horizon_s is required"):
        load_instance(text_file)


def test_a_missing_file_is_a_document_error(tmp_path: Path) -> None:
    with pytest.raises(DocumentError, match="cannot read"):
        load_instance(tmp_path / "absent.yaml")
