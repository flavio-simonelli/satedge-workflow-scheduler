"""Command-line interface of the scheduler."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. The help text is user-facing and in Italian."""
    parser = argparse.ArgumentParser(
        prog="satedge-workflow-scheduler",
        description="Pianificazione delle richieste di workflow sulla costellazione.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the CLI. Returns the exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
