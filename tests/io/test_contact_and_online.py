"""The contact plan and the online state of a planning epoch, read with the instance."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from satedge_workflow_scheduler.domain.network import WindowedAvailability
from satedge_workflow_scheduler.io import DocumentError, load_instance
from satedge_workflow_scheduler.io.documents import ResultDoc
from satedge_workflow_scheduler.io.results import write_result
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.run import SolverRun
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig
from satedge_workflow_scheduler.solvers.sequential.solver import SequentialSolver

# S2-G has no windows here: the contact plan gives them
OPEN = """\
schema_version: "1.0.0"
ground_station: G
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
    - {a: S2, b: G, capacity_gbps: 2, latency_s: 1}
requests:
  - id: r1
    priority: urgent
    origin: S1
    release_s: 0
    raw_size_gb: 2
    tasks:
      - {execution_s: 10, output_size_gb: 0.5}
  - id: r2
    priority: non_urgent
    origin: S2
    release_s: 0
    raw_size_gb: 2
    tasks:
      - {execution_s: 10, output_size_gb: 0.5}
"""

PLAN = """\
schema_version: "1.0.0"
links:
  - a: G
    b: S2
    windows: [{start_s: 100, end_s: 200}]
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_the_contact_plan_makes_the_named_link_windowed(tmp_path: Path) -> None:
    instance = load_instance(
        _write(tmp_path, "instance.yaml", OPEN),
        contact_plan=_write(tmp_path, "contact.yaml", PLAN),
    )
    to_ground = next(a for a in instance.network.arcs if a.target == "G")
    assert isinstance(to_ground.availability, WindowedAvailability)
    assert [w.start for w in to_ground.availability.windows.windows] == [100.0]
    # the link S1-S2 is not in the plan: it stays always available
    s1_s2 = next(a for a in instance.network.arcs if a.source == "S1")
    assert not isinstance(s1_s2.availability, WindowedAvailability)


def test_a_contact_plan_for_a_missing_link_is_rejected(tmp_path: Path) -> None:
    plan = PLAN.replace("a: G", "a: S1").replace("b: S2", "b: G")
    instance_file = _write(tmp_path, "instance.yaml", OPEN)
    contact_file = _write(tmp_path, "contact.yaml", plan)
    with pytest.raises(DocumentError, match="no link S1-G"):
        load_instance(instance_file, contact_plan=contact_file)


def test_a_link_windowed_in_both_files_is_rejected(tmp_path: Path) -> None:
    windowed = OPEN.replace(
        "- {a: S2, b: G, capacity_gbps: 2, latency_s: 1}",
        "- {a: S2, b: G, capacity_gbps: 2, latency_s: 1, windows: [{start_s: 0, end_s: 5}]}",
    )
    instance_file = _write(tmp_path, "instance.yaml", windowed)
    contact_file = _write(tmp_path, "contact.yaml", PLAN)
    with pytest.raises(DocumentError, match="windowed in the instance"):
        load_instance(instance_file, contact_plan=contact_file)


def test_the_online_state_gives_plan_release_and_inline_committed_calendar(
    tmp_path: Path,
) -> None:
    online = _write(
        tmp_path,
        "online.yaml",
        """\
schema_version: "1.0.0"
plan_release_s: {S2: 50.5, S1: null}
committed:
  executions:
    - {request: r1, task: 0, satellite: S1, start_s: 0, end_s: 10}
""",
    )
    instance = load_instance(_write(tmp_path, "instance.yaml", OPEN), online=online)

    assert instance.plan_release["S2"] == pytest.approx(50.5)
    assert math.isinf(instance.plan_release["S1"])
    assert instance.committed.request_ids == frozenset({"r1"})
    assert instance.is_committed("r1")
    assert not instance.is_committed("r2")


def test_the_committed_calendar_comes_from_a_saved_result_and_its_filter(
    tmp_path: Path,
) -> None:
    instance = load_instance(_write(tmp_path, "instance.yaml", OPEN))
    run: SolverRun = SequentialSolver(SequentialConfig()).run(instance)
    write_result(tmp_path / "results" / "epoch1.json", instance, run, DeadlineMode.HARD)
    assert ResultDoc.model_validate_json(
        (tmp_path / "results" / "epoch1.json").read_text(encoding="utf-8")
    )

    online = _write(
        tmp_path,
        "online.yaml",
        """\
schema_version: "1.0.0"
committed:
  result: results/epoch1.json
  requests: [r2]
""",
    )
    epoch = load_instance(_write(tmp_path, "instance.yaml", OPEN), online=online)
    assert epoch.committed.request_ids == frozenset({"r2"})
    assert all(e.request_id == "r2" for e in epoch.committed.executions)


def test_a_committed_block_with_both_result_and_inline_is_rejected(
    tmp_path: Path,
) -> None:
    online = _write(
        tmp_path,
        "online.yaml",
        """\
schema_version: "1.0.0"
committed:
  result: epoch1.json
  executions:
    - {request: r1, task: 0, satellite: S1, start_s: 0, end_s: 10}
""",
    )
    instance_file = _write(tmp_path, "instance.yaml", OPEN)
    with pytest.raises(DocumentError, match="either 'result' or inline"):
        load_instance(instance_file, online=online)
