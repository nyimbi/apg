#!/usr/bin/env python3
"""APG lint command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from compiler.linting import audit_lint_fixtures, lint_path


console = Console()


def _print_text_report(report: dict[str, Any]) -> None:
	if not report["files"]:
		console.print("[red]No APG source files found[/red]")
		return

	status = "[green]OK[/green]" if report["ok"] else "[red]FAILED[/red]"
	counts = report["severity_counts"]
	console.print(
		f"APG lint {status}: {len(report['files'])} file(s), "
		f"{counts['error']} error(s), {counts['warning']} warning(s)"
	)
	for diagnostic in report["diagnostics"]:
		start = diagnostic["range"]["start"]
		console.print(
			f"  {diagnostic['file']}:{start['line'] + 1}:{start['character']}: "
			f"{diagnostic['code']} {diagnostic['severity']}: {diagnostic['message']}"
		)


@click.command()
@click.argument("path", required=False, type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit apg.lint-report.v1 JSON")
@click.option("--strict", is_flag=True, help="Treat warnings as errors")
@click.option("--catalog", type=click.Path(path_type=Path), default=None, help="Capability contract root or local apg.capability-catalog.v1 file")
@click.option("--audit-fixtures", is_flag=True, help="Audit checked-in lint fixtures")
@click.option("--fixtures", type=click.Path(path_type=Path), default=None, help="Lint fixture catalog path")
def lint(
	path: Path | None,
	as_json: bool,
	strict: bool,
	catalog: Path | None,
	audit_fixtures: bool,
	fixtures: Path | None,
) -> None:
	"""Lint APG source without writing generated code."""
	if audit_fixtures:
		if path is not None or catalog is not None or strict:
			raise click.ClickException("--audit-fixtures cannot be combined with PATH, --catalog, or --strict")
		report = audit_lint_fixtures(fixtures)
		if as_json:
			click.echo(json.dumps(report, indent=2, sort_keys=True))
		else:
			summary = report["summary"]
			status = "OK" if report["ok"] else "FAILED"
			click.echo(
				f"APG lint fixtures {status}: "
				f"{summary['passing_fixture_count']}/{summary['fixture_count']} passing, "
				f"{summary['blocking_gap_count']} blocking gaps"
			)
		if not report["ok"]:
			raise click.exceptions.Exit(1)
		return

	if path is None:
		raise click.ClickException("Specify an APG path or use --audit-fixtures")
	if catalog is not None and not catalog.exists():
		raise click.ClickException(f"Capability catalog not found: {catalog}")
	if not path.exists():
		raise click.ClickException(f"APG path not found: {path}")

	report = lint_path(path, strict=strict, catalog=catalog)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		_print_text_report(report)

	if not report["ok"]:
		raise click.exceptions.Exit(1)
