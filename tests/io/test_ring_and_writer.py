"""The ring constellation, and the instance files written back: they read back unchanged."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    WindowedAvailability,
)
from satedge_workflow_scheduler.io import (
    DocumentError,
    load_instance,
    load_online_state,
)
from satedge_workflow_scheduler.io.files import load_instance as load
from satedge_workflow_scheduler.io.writer import write_instance, write_online_state
from satedge_workflow_scheduler.solvers.run import SolverRun
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig
from satedge_workflow_scheduler.solvers.sequential.solver import SequentialSolver

RING = """\
schema_version: "1.0.0"
name: ring
horizon_s: 300
topology:
  kind: ring_constellation
  planes: 2
  satellites_per_plane: 3
  com_index: 0
  plmm_capacity_gb: 20
  plmm_overrides_gb: {COM1: 15}
  hpc: {period_s: 100, duration_s: 30, startup_s: 5}
  intra_link: {capacity_gbps: 1, latency_s: 0.5}
requests:
  - id: r1
    priority: urgent
    origin: P1-S1
    release_s: 0
    raw_size_gb: 2
    tasks:
      - {execution_s: 10, output_size_gb: 0.5}
"""

CONTACT = """\
schema_version: "1.0.0"
links:
  - {a: COM1, b: COM2, windows: [{start_s: 50, end_s: 150}]}
  - {a: COM1, b: GS, windows: [{start_s: 20, end_s: 80}]}
  - {a: COM2, b: GS, windows: [{start_s: 90, end_s: 120}]}
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _ring(tmp_path: Path) -> Path:
    return _write(tmp_path, "ring.yaml", RING)


def test_the_ring_has_the_named_nodes_and_its_links(tmp_path: Path) -> None:
    instance = load(
        _ring(tmp_path),
        contact_plan=_write(tmp_path, "contact.yaml", CONTACT),
    )
    names = [s.id for s in instance.network.satellites]
    assert names == ["COM1", "P1-S1", "P1-S2", "COM2", "P2-S1", "P2-S2"]
    # 3 links per plane (a cycle of three), one COM-COM link and two COM-ground links
    pairs = {frozenset((a.source, a.target)) for a in instance.network.arcs}
    assert len(pairs) == 3 * 2 + 1 + 2


def test_the_ring_windows_come_from_the_contact_plan(tmp_path: Path) -> None:
    instance = load(
        _ring(tmp_path),
        contact_plan=_write(tmp_path, "contact.yaml", CONTACT),
    )
    com1_com2 = next(
        a for a in instance.network.arcs if {a.source, a.target} == {"COM1", "COM2"}
    )
    assert isinstance(com1_com2.availability, WindowedAvailability)
    intra = next(
        a for a in instance.network.arcs if {a.source, a.target} == {"P1-S1", "P1-S2"}
    )
    assert isinstance(intra.availability, AlwaysAvailable)


def test_a_windowed_ring_link_without_contact_windows_is_rejected(
    tmp_path: Path,
) -> None:
    ring = _ring(tmp_path)
    with pytest.raises(DocumentError, match="its windows come from the contact plan"):
        load(ring)


def test_the_hpc_windows_repeat_with_the_period_and_the_startup_is_taken_off(
    tmp_path: Path,
) -> None:
    instance = load(_ring(tmp_path), contact_plan=_write(tmp_path, "c.yaml", CONTACT))
    sat = instance.network.satellite("P1-S1")
    windows = sat.hpc_windows.windows
    # index 1 of the 6 satellites: phase = 1 * 100 / 6; each window lasts 30 - 5 seconds
    assert windows[0].start == pytest.approx(100 / 6)
    assert windows[0].duration == pytest.approx(25.0)
    assert windows[1].start - windows[0].start == pytest.approx(100.0)


def test_the_plmm_override_applies_to_its_satellite_only(tmp_path: Path) -> None:
    instance = load(_ring(tmp_path), contact_plan=_write(tmp_path, "c.yaml", CONTACT))
    assert instance.network.satellite("COM1").plmm_capacity == pytest.approx(15.0)
    assert instance.network.satellite("COM2").plmm_capacity == pytest.approx(20.0)


def test_an_override_for_a_missing_satellite_is_rejected(tmp_path: Path) -> None:
    text = RING.replace("{COM1: 15}", "{COM9: 15}")
    ring_file = _write(tmp_path, "ring.yaml", text)
    contact_file = _write(tmp_path, "c.yaml", CONTACT)
    with pytest.raises(DocumentError, match="COM9 is not a satellite"):
        load(ring_file, contact_plan=contact_file)


def test_an_instance_written_and_read_back_is_the_same(tmp_path: Path) -> None:
    original = load(_ring(tmp_path), contact_plan=_write(tmp_path, "c.yaml", CONTACT))
    file = tmp_path / "out" / "epoch.yaml"
    write_instance(file, original)
    again = load_instance(file)

    assert again.name == original.name
    assert again.horizon == pytest.approx(original.horizon)
    assert again.requests == original.requests
    assert [s.id for s in again.network.satellites] == [
        s.id for s in original.network.satellites
    ]
    assert again.network.ground_station == original.network.ground_station
    assert _arc_keys(again) == _arc_keys(original)
    assert dict(again.plan_release) == pytest.approx(dict(original.plan_release))


def _arc_keys(instance: ProblemInstance) -> list[tuple[object, ...]]:
    keys = []
    for arc in instance.network.arcs:
        windows: tuple[tuple[float, float], ...] = ()
        if isinstance(arc.availability, WindowedAvailability):
            windows = tuple((w.start, w.end) for w in arc.availability.windows.windows)
        keys.append((arc.source, arc.target, arc.capacity, arc.latency, windows))
    return sorted(keys, key=str)


def test_the_epoch_state_written_and_read_back_is_the_same(tmp_path: Path) -> None:
    instance = load(_ring(tmp_path), contact_plan=_write(tmp_path, "c.yaml", CONTACT))
    run: SolverRun = SequentialSolver(SequentialConfig()).run(instance)
    release = {**instance.plan_release, "COM2": 40.0}
    epoch = dataclasses.replace(
        instance, plan_release=release, committed=run.solution.schedule
    )
    file = tmp_path / "online.yaml"
    write_online_state(file, epoch)
    read_release, schedule = load_online_state(file)

    assert schedule == epoch.committed
    assert read_release["COM2"] == pytest.approx(40.0)
