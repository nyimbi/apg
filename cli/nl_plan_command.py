#!/usr/bin/env python3
"""APG natural-language plan command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.nl_plan import audit_nl_plan_fixtures, build_nl_plan


@click.command(name="nl-plan")
@click.argument("source_file", required=False, type=click.Path(path_type=Path))
@click.option("--audit-fixtures", is_flag=True, help="Audit checked-in natural-language planner fixtures")
@click.option("--fixtures", type=click.Path(path_type=Path), default=None, help="Natural-language planner fixture catalog path")
@click.option("--prompt", help="Natural-language APG edit request")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.nl-plan.v1 JSON")
def nl_plan(source_file: Path | None, audit_fixtures: bool, fixtures: Path | None, prompt: str | None, as_json: bool) -> None:
	"""Plan a bounded APG DSL patch without mutating source or generating code."""
	if audit_fixtures:
		if source_file is not None or prompt:
			raise click.ClickException("--audit-fixtures cannot be combined with a source file or --prompt")
		report = audit_nl_plan_fixtures(fixtures)
		if as_json:
			click.echo(json.dumps(report, indent=2, sort_keys=True))
		else:
			summary = report["summary"]
			status = "OK" if report["ok"] else "FAILED"
			click.echo(
				f"APG nl-plan fixtures {status}: "
				f"{summary['passing_fixture_count']}/{summary['fixture_count']} passing, "
				f"{summary['blocking_gap_count']} blocking gaps"
			)
		if not report["ok"]:
			raise click.exceptions.Exit(1)
		return
	if source_file is None:
		raise click.ClickException("Specify an APG source file or use --audit-fixtures")
	if not prompt:
		raise click.ClickException("--prompt is required unless --audit-fixtures is used")
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG nl-plan expects a file: {source_file}")

	report = build_nl_plan(source_file, prompt)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(
			f"APG nl-plan {status}: {source_file}, "
			f"intent={report['intent']}, symbols={len(report['affected_symbols'])}"
		)
		if report["dsl_patch"]:
			click.echo(report["dsl_patch"])
		for warning in report["warnings"]:
			click.echo(f"  warning: {warning}")
		for error in report["errors"]:
			click.echo(f"  error: {error}")

	if not report["ok"]:
		raise click.exceptions.Exit(1)
