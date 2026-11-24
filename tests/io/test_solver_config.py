"""The solver file: the configuration it describes, and the errors it reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from satedge_workflow_scheduler.io import DocumentError
from satedge_workflow_scheduler.io.files import load_solver_config
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.config import LazyGroup, Level, MilpConfig
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig

MILP = """\
schema_version: "1.0.0"
method: milp
deadline: soft
milp:
  level: 6
  lazy: [plmm]
  strengthening: {arrival_without_big_m: true}
limits:
  time_limit_s: 60
  mip_gap: 0.0
  threads: 2
verbose: true
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "solver.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_milp_file_gives_the_milp_configuration(tmp_path: Path) -> None:
    config = load_solver_config(_write(tmp_path, MILP))

    assert isinstance(config, MilpConfig)
    assert config.level is Level.PLMM_CAPACITY
    assert config.deadline is DeadlineMode.SOFT
    assert config.lazy == frozenset({LazyGroup.PLMM})
    assert config.strengthening.arrival_without_big_m is True
    assert config.solve.time_limit == 60.0
    assert config.solve.mip_gap == 0.0
    assert config.solve.threads == 2
    assert config.solve.verbose is True


def test_the_sequential_file_gives_the_heuristic_configuration(tmp_path: Path) -> None:
    text = (
        'schema_version: "1.0.0"\nmethod: sequential\nsequential: {max_backtracks: 7}\n'
    )
    config = load_solver_config(_write(tmp_path, text))

    assert isinstance(config, SequentialConfig)
    assert config.max_backtracks == 7
    assert config.deadline is DeadlineMode.HARD


def test_a_level_that_does_not_exist_is_a_document_error(tmp_path: Path) -> None:
    text = MILP.replace("level: 6", "level: 2")
    level_file = _write(tmp_path, text)
    with pytest.raises(DocumentError, match="milp.level"):
        load_solver_config(level_file)


def test_the_milp_method_needs_its_section(tmp_path: Path) -> None:
    text = 'schema_version: "1.0.0"\nmethod: milp\n'
    section_file = _write(tmp_path, text)
    with pytest.raises(DocumentError, match="needs a 'milp' section"):
        load_solver_config(section_file)


def test_an_unknown_lazy_group_is_rejected(tmp_path: Path) -> None:
    text = MILP.replace("lazy: [plmm]", "lazy: [colour]")
    colour_file = _write(tmp_path, text)
    with pytest.raises(DocumentError, match="colour"):
        load_solver_config(colour_file)


def test_an_unknown_method_is_rejected_with_its_field(tmp_path: Path) -> None:
    method_file = _write(tmp_path, 'schema_version: "1.0.0"\nmethod: genetic\n')
    with pytest.raises(DocumentError, match=r"method"):
        load_solver_config(method_file)


def test_limits_are_rejected_for_the_sequential_heuristic(tmp_path: Path) -> None:
    text = 'schema_version: "1.0.0"\nmethod: sequential\nlimits: {time_limit_s: 10}\n'
    limits_file = _write(tmp_path, text)
    with pytest.raises(DocumentError, match="limits apply to method 'milp' only"):
        load_solver_config(limits_file)
