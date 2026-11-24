"""Command-line interface: solve an instance, check a result, compare solvers, write schemas.

Exit codes: 0 on success; 1 when the data is invalid, a solve fails, or a check finds violations;
2 when the arguments are wrong (reported by argparse).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import BaseModel

from satedge_workflow_scheduler import api, report
from satedge_workflow_scheduler.common.errors import SchedulerError
from satedge_workflow_scheduler.io.documents import (
    ContactPlanDoc,
    InstanceDoc,
    OnlineDoc,
    ResultDoc,
    SolverDoc,
)
from satedge_workflow_scheduler.io.files import load_instance, load_solver_config
from satedge_workflow_scheduler.io.reading import read_document
from satedge_workflow_scheduler.io.results import solution_from_document
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.feasibility import check
from satedge_workflow_scheduler.solvers.milp.config import Level, MilpConfig
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig

SCHEMAS: dict[str, type[BaseModel]] = {
    "instance": InstanceDoc,
    "contact_plan": ContactPlanDoc,
    "online": OnlineDoc,
    "solver": SolverDoc,
    "result": ResultDoc,
}
"""The documents whose JSON Schema the ``schema`` command writes."""


class UsageError(SchedulerError):
    """The options given to a command do not fit together."""


def build_parser() -> argparse.ArgumentParser:
    """The parser of the command line."""
    parser = argparse.ArgumentParser(
        prog="satedge-workflow-scheduler",
        description="Pianificazione delle richieste di workflow sulla costellazione.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMANDO")

    solve = commands.add_parser("solve", help="risolve un'istanza")
    solve.add_argument("instance", type=Path, help="file YAML dell'istanza")
    _instance_options(solve)
    solve.add_argument("-s", "--solver", type=Path, help="file YAML del solver")
    solve.add_argument(
        "-o", "--output", type=Path, help="file JSON in cui salvare il risultato"
    )
    solve.add_argument(
        "--method", choices=["milp", "sequential"], help="metodo di soluzione"
    )
    solve.add_argument(
        "--deadline", choices=["hard", "soft"], help="modalità delle scadenze"
    )
    solve.add_argument(
        "--level", type=int, choices=[3, 4, 5, 6], help="livello del MILP"
    )
    solve.add_argument(
        "--time-limit", type=float, metavar="S", help="limite di tempo del MILP"
    )
    solve.add_argument("--mip-gap", type=float, help="gap relativo del MILP")
    solve.add_argument("--threads", type=int, help="thread del MILP")
    solve.add_argument(
        "--check", action="store_true", help="esce con 1 se ci sono violazioni"
    )
    solve.add_argument(
        "-v", "--verbose", action="store_true", help="mostra il log del MILP"
    )
    solve.add_argument(
        "-q", "--quiet", action="store_true", help="non stampa il riepilogo"
    )
    solve.set_defaults(handler=_solve)

    check_cmd = commands.add_parser("check", help="verifica un risultato sull'istanza")
    check_cmd.add_argument("instance", type=Path, help="file YAML dell'istanza")
    check_cmd.add_argument("result", type=Path, help="file JSON del risultato")
    _instance_options(check_cmd)
    check_cmd.set_defaults(handler=_check)

    compare = commands.add_parser(
        "compare", help="confronta più solutori sulla stessa istanza"
    )
    compare.add_argument("instance", type=Path, help="file YAML dell'istanza")
    compare.add_argument(
        "solvers", type=Path, nargs="+", metavar="SOLVER", help="file YAML dei solutori"
    )
    _instance_options(compare)
    compare.add_argument(
        "-o", "--output", type=Path, help="file .csv o .json con il confronto"
    )
    compare.set_defaults(handler=_compare)

    schema = commands.add_parser("schema", help="scrive i JSON Schema dei file")
    schema.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("schemas"),
        help="cartella di destinazione",
    )
    schema.set_defaults(handler=_schema)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the exit code."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except SchedulerError as error:
        print(f"errore: {error}", file=sys.stderr)
        return 1


def _instance_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--contact-plan", type=Path, help="file YAML del contact plan"
    )
    parser.add_argument("--online", type=Path, help="file YAML dello stato online")


def _solve(args: argparse.Namespace) -> int:
    instance = load_instance(
        args.instance, contact_plan=args.contact_plan, online=args.online
    )
    result = api.solve(instance, _solver_config(args))
    if args.output is not None:
        result.save(args.output)
    if not args.quiet:
        print(report.summary(result))
    if args.check and result.metrics.violations:
        print(report.violations_text(result.metrics.violations), file=sys.stderr)
        return 1
    return 0


def _check(args: argparse.Namespace) -> int:
    instance = load_instance(
        args.instance, contact_plan=args.contact_plan, online=args.online
    )
    document = read_document(args.result, ResultDoc)
    violations = check(
        instance, solution_from_document(document), DeadlineMode(document.deadline)
    )
    if violations:
        print(report.violations_text(violations), file=sys.stderr)
        return 1
    print("risultato valido")
    return 0


def _compare(args: argparse.Namespace) -> int:
    instance = load_instance(
        args.instance, contact_plan=args.contact_plan, online=args.online
    )
    configs: dict[str, api.SolverConfig] = {}
    for path in args.solvers:
        label = path.stem
        if label in configs:
            raise UsageError(f"due solutori hanno lo stesso nome: {label}")
        configs[label] = load_solver_config(path)
    results = api.compare(instance, configs)
    rows = report.comparison_rows(results)
    print(report.comparison_table(rows))
    if args.output is not None:
        report.write_comparison(args.output, rows)
    return 1 if any(row["violazioni"] for row in rows) else 0


def _schema(args: argparse.Namespace) -> int:
    directory: Path = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMAS.items():
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        file = directory / f"{name}.schema.json"
        file.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(file)
    return 0


def _solver_config(args: argparse.Namespace) -> api.SolverConfig | None:
    """The configuration of ``solve``: the solver file, with the options given on the line."""
    base: api.SolverConfig | None = (
        None if args.solver is None else load_solver_config(args.solver)
    )
    method = args.method
    if method is None and base is not None:
        method = "sequential" if isinstance(base, SequentialConfig) else "milp"
    if method is None:
        return None if not _overrides_given(args) else _milp(args, None)
    if method == "sequential":
        return _sequential(args, base)
    return _milp(args, base)


def _sequential(
    args: argparse.Namespace, base: api.SolverConfig | None
) -> SequentialConfig:
    if _milp_options_given(args):
        raise UsageError(
            "--level, --time-limit, --mip-gap, --threads e -v valgono solo per il MILP"
        )
    config = base if isinstance(base, SequentialConfig) else SequentialConfig()
    if args.deadline is not None:
        config = dataclasses.replace(config, deadline=DeadlineMode(args.deadline))
    return config


def _milp(args: argparse.Namespace, base: api.SolverConfig | None) -> MilpConfig:
    config = (
        base if isinstance(base, MilpConfig) else MilpConfig(level=Level.PLMM_CAPACITY)
    )
    if args.level is not None:
        config = dataclasses.replace(config, level=Level(args.level))
    if args.deadline is not None:
        config = dataclasses.replace(config, deadline=DeadlineMode(args.deadline))
    solve = config.solve
    if args.time_limit is not None:
        solve = dataclasses.replace(solve, time_limit=args.time_limit)
    if args.mip_gap is not None:
        solve = dataclasses.replace(solve, mip_gap=args.mip_gap)
    if args.threads is not None:
        solve = dataclasses.replace(solve, threads=args.threads)
    if args.verbose:
        solve = dataclasses.replace(solve, verbose=True)
    return dataclasses.replace(config, solve=solve)


def _overrides_given(args: argparse.Namespace) -> bool:
    return _milp_options_given(args) or args.deadline is not None


def _milp_options_given(args: argparse.Namespace) -> bool:
    return any(
        value is not None and value is not False
        for value in (
            args.level,
            args.time_limit,
            args.mip_gap,
            args.threads,
            args.verbose,
        )
    )


def _version() -> str:
    try:
        return version("satedge-workflow-scheduler")
    except PackageNotFoundError:
        return "0+unknown"
