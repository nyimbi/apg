#!/usr/bin/env python3
"""APG semantic model command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.semantic_model import audit_semantic_model_fixtures, build_semantic_model


@click.command(name="model")
@click.argument("source_file", required=False, type=click.Path(path_type=Path))
@click.option("--audit-fixtures", is_flag=True, help="Audit checked-in semantic-model fixtures")
@click.option("--fixtures", type=click.Path(path_type=Path), default=None, help="Semantic-model fixture catalog path")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.semantic-model.v1 JSON")
def model(source_file: Path | None, audit_fixtures: bool, fixtures: Path | None, as_json: bool) -> None:
	"""Emit the normalized semantic model for APG source."""
	if audit_fixtures:
		if source_file is not None:
			raise click.ClickException("--audit-fixtures cannot be combined with SOURCE_FILE")
		report = audit_semantic_model_fixtures(fixtures)
		if as_json:
			click.echo(json.dumps(report, indent=2, sort_keys=True))
		else:
			summary = report["summary"]
			status = "OK" if report["ok"] else "FAILED"
			click.echo(
				f"APG semantic-model fixtures {status}: "
				f"{summary['passing_fixture_count']}/{summary['fixture_count']} passing, "
				f"{summary['blocking_gap_count']} blocking gaps"
			)
		if not report["ok"]:
			raise click.exceptions.Exit(1)
		return
	if source_file is None:
		raise click.ClickException("Specify an APG source file or use --audit-fixtures")
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG model expects a file: {source_file}")

	report = build_semantic_model(source_file)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(
			f"APG model {status}: {source_file}, "
			f"{len(report['symbols'])} symbol(s), "
			f"{len(report['tables'])} table(s), "
			f"{len(report['agents'])} agent(s), "
			f"{len(report['capabilities'])} capability(ies)"
		)

	if not report["ok"]:
		raise click.exceptions.Exit(1)
