#!/usr/bin/env python3
"""APG semantic drift command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.drift import audit_drift_fixtures, build_drift_report


@click.command(name="drift")
@click.argument("source_file", required=False, type=click.Path(path_type=Path))
@click.option("--audit-fixtures", is_flag=True, help="Audit semantic drift fixture catalog")
@click.option("--fixtures", type=click.Path(path_type=Path), default=None, help="Semantic drift fixture directory")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
def drift(source_file: Path | None, audit_fixtures: bool, fixtures: Path | None, as_json: bool) -> None:
	"""Detect semantic drift between compiler, generated artifact, and runtime surfaces."""
	if audit_fixtures:
		report = audit_drift_fixtures(fixtures)
		if as_json:
			click.echo(json.dumps(report, indent=2, sort_keys=True))
		else:
			status = "OK" if report["ok"] else "FAILED"
			summary = report["summary"]
			click.echo(
				f"APG drift fixtures {status}: "
				f"{summary['passed']}/{summary['fixture_count']} passed"
			)
			for error in report["errors"]:
				click.echo(f"  error: {error}")
		if not report["ok"]:
			raise click.exceptions.Exit(1)
		return

	if source_file is None:
		raise click.ClickException("Specify an APG source file or use --audit-fixtures")
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG drift expects a file: {source_file}")

	report = build_drift_report(source_file)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		summary = report["summary"]
		click.echo(
			f"APG drift {status}: {source_file}, "
			f"{summary['comparison_count']} comparison(s), "
			f"{summary['drift_count']} drift(s)"
		)
		for error in report["errors"]:
			click.echo(f"  error: {error}")

	if not report["ok"]:
		raise click.exceptions.Exit(1)
