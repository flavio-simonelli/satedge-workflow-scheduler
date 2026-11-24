"""Text of the results: the summary of a solve, the violations, and the comparison of solvers."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from satedge_workflow_scheduler.api import Result
from satedge_workflow_scheduler.domain.requests import Priority
from satedge_workflow_scheduler.domain.solution import Solution
from satedge_workflow_scheduler.scheduling.feasibility import Violation


def summary(result: Result) -> str:
    """The lines written after a solve, in Italian."""
    stats = result.statistics
    metrics = result.metrics
    solution = result.solution
    lines = [
        f"Istanza: {result.instance.name or '(senza nome)'}",
        f"Metodo: {stats.method}; scadenze: {result.deadline.value}",
        f"Stato: {solution.status.value}",
        f"Richieste accettate: {len(metrics.accepted_ids)} su {len(metrics.requests)}",
        _by_priority(result, Priority.URGENT, "urgenti"),
        _by_priority(result, Priority.NON_URGENT, "non urgenti"),
        f"Ritardo totale: {metrics.total_tardiness:.1f} s",
        f"Tempo di risposta medio: {_seconds(metrics.mean_response)}",
        f"Obiettivo: {_objective(solution)}",
        f"Tempo del solver: {_seconds(stats.solver_time_s)}; tempo totale: {_seconds(stats.wall_time_s)}",
        # the bound of the solver is on the objective of the model, not on the one above:
        # it stays in the result file, and the gap says whether the solution is optimal
        f"Gap: {_number(stats.gap)}",
        f"Violazioni: {len(metrics.violations)}",
    ]
    return "\n".join(lines)


def violations_text(violations: Sequence[Violation]) -> str:
    """One line per violation, in Italian-friendly plain text."""
    return "\n".join(f"violazione: {v}" for v in violations)


def comparison_rows(results: Mapping[str, Result]) -> list[dict[str, object]]:
    """One row per solver, with the figures to compare them."""
    rows: list[dict[str, object]] = []
    for label, result in results.items():
        metrics = result.metrics
        solution = result.solution
        rows.append(
            {
                "solutore": label,
                "metodo": result.statistics.method,
                "stato": solution.status.value,
                "accettate": len(metrics.accepted_ids),
                "rifiutate": len(metrics.requests) - len(metrics.accepted_ids),
                "ritardo_totale_s": round(metrics.total_tardiness, 3),
                "risposta_media_s": _round(metrics.mean_response),
                "obiettivo": _round(
                    solution.objective.primary if solution.objective else None
                ),
                "tempo_solver_s": _round(result.statistics.solver_time_s),
                "tempo_totale_s": _round(result.statistics.wall_time_s),
                "gap": _round(result.statistics.gap),
                "violazioni": len(metrics.violations),
            }
        )
    return rows


def comparison_table(rows: Sequence[Mapping[str, object]]) -> str:
    """The rows as an aligned table."""
    if not rows:
        return "nessun solutore"
    columns = list(rows[0])
    cells = [[_cell(row[c]) for c in columns] for row in rows]
    widths = [
        max(len(c), *(len(line[i]) for line in cells)) for i, c in enumerate(columns)
    ]
    header = "  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True))
    body = [
        "  ".join(v.ljust(w) for v, w in zip(line, widths, strict=True))
        for line in cells
    ]
    return "\n".join([header, *body])


def write_comparison(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write the rows as CSV, or as JSON if the file name does not end in ``.csv``."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.suffix.lower() == ".csv":
        with file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    else:
        file.write_text(
            json.dumps(list(rows), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _by_priority(result: Result, priority: Priority, label: str) -> str:
    metrics = result.metrics
    accepted = metrics.count(priority, accepted=True)
    total = accepted + metrics.count(priority, accepted=False)
    return f"  {label}: {accepted} su {total}"


def _objective(solution: Solution) -> str:
    if solution.objective is None:
        return "-"
    return f"{solution.objective.primary:.4g} (risposta {solution.objective.response:.1f} s)"


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f} s"


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _cell(value: object) -> str:
    return "-" if value is None else str(value)
