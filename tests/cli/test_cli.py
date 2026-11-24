"""The command line: each command on files in a temporary folder, with its exit code."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.cli import main
from satedge_workflow_scheduler.io import write_instance

SEQUENTIAL = 'schema_version: "1.0.0"\nmethod: sequential\n'


@pytest.fixture
def files(tmp_path: Path) -> Path:
    write_instance(tmp_path / "instance.yaml", sample_instance())
    (tmp_path / "seq.yaml").write_text(SEQUENTIAL, encoding="utf-8")
    (tmp_path / "seq2.yaml").write_text(SEQUENTIAL, encoding="utf-8")
    return tmp_path


def test_the_version_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "satedge-workflow-scheduler" in capsys.readouterr().out


def test_solve_prints_the_summary_and_saves_the_result(
    files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "solve",
            str(files / "instance.yaml"),
            "-s",
            str(files / "seq.yaml"),
            "-o",
            str(files / "out" / "result.json"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Richieste accettate: 2 su 2" in out
    assert (
        json.loads((files / "out" / "result.json").read_text())["statistics"]["method"]
        == "sequential"
    )


def test_quiet_solve_prints_nothing(
    files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["solve", str(files / "instance.yaml"), "-s", str(files / "seq.yaml"), "-q"]
    )
    assert code == 0
    assert capsys.readouterr().out == ""


def test_a_milp_option_with_the_heuristic_is_a_usage_error(
    files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "solve",
            str(files / "instance.yaml"),
            "--method",
            "sequential",
            "--level",
            "5",
        ]
    )
    assert code == 1
    assert "valgono solo per il MILP" in capsys.readouterr().err


def test_a_missing_instance_is_reported_with_exit_code_one(
    files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["solve", str(files / "absent.yaml"), "--method", "sequential"])
    assert code == 1
    assert capsys.readouterr().err.startswith("errore: ")


def test_an_invalid_instance_names_the_field(
    files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = files / "bad.yaml"
    bad.write_text(
        (files / "instance.yaml")
        .read_text(encoding="utf-8")
        .replace("release_s: 0", "release_s: soon", 1),
        encoding="utf-8",
    )
    code = main(["solve", str(bad), "--method", "sequential"])
    assert code == 1
    assert "release_s" in capsys.readouterr().err


def test_check_accepts_a_valid_result_and_rejects_a_tampered_one(
    files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "solve",
            str(files / "instance.yaml"),
            "-s",
            str(files / "seq.yaml"),
            "-q",
            "-o",
            str(files / "result.json"),
        ]
    )
    assert (
        main(["check", str(files / "instance.yaml"), str(files / "result.json")]) == 0
    )
    assert "risultato valido" in capsys.readouterr().out

    data = json.loads((files / "result.json").read_text(encoding="utf-8"))
    data["executions"][0]["end_s"] = 1e6  # the task now runs far outside its window
    (files / "tampered.json").write_text(json.dumps(data), encoding="utf-8")
    assert (
        main(["check", str(files / "instance.yaml"), str(files / "tampered.json")]) == 1
    )
    assert "violazione" in capsys.readouterr().err


def test_compare_writes_one_row_per_solver_in_csv(
    files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "compare",
            str(files / "instance.yaml"),
            str(files / "seq.yaml"),
            str(files / "seq2.yaml"),
            "-o",
            str(files / "cmp.csv"),
        ]
    )
    assert code == 0
    assert "seq" in capsys.readouterr().out
    with (files / "cmp.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["solutore"] for row in rows] == ["seq", "seq2"]
    assert rows[0]["accettate"] == "2"


def test_compare_rejects_two_solvers_with_the_same_name(
    files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (files / "a").mkdir()
    (files / "a" / "seq.yaml").write_text(SEQUENTIAL, encoding="utf-8")
    code = main(
        [
            "compare",
            str(files / "instance.yaml"),
            str(files / "seq.yaml"),
            str(files / "a" / "seq.yaml"),
        ]
    )
    assert code == 1
    assert "lo stesso nome" in capsys.readouterr().err


def test_schema_writes_the_json_schema_of_each_document(files: Path) -> None:
    assert main(["schema", "-o", str(files / "schemas")]) == 0
    for name in ("instance", "contact_plan", "online", "solver", "result"):
        schema = json.loads(
            (files / "schemas" / f"{name}.schema.json").read_text(encoding="utf-8")
        )
        assert schema["$schema"].endswith("2020-12/schema")
        assert "properties" in schema or "$defs" in schema


def test_wrong_arguments_exit_with_two(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["solve"])
    assert exit_info.value.code == 2
