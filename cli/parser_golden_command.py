#!/usr/bin/env python3
"""APG parser golden audit command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.parser_golden import audit_parser_golden


@click.command(name="parser-golden")
@click.option("--catalog", type=click.Path(path_type=Path), default=None, help="Parser golden catalog path")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.parser-golden-audit.v1 JSON")
def parser_golden(catalog: Path | None, as_json: bool) -> None:
	"""Audit parser golden fixtures and required grammar construct coverage."""
	report = audit_parser_golden(catalog)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		summary = report["summary"]
		click.echo(
			f"APG parser-golden {status}: "
			f"{summary['passing_fixture_count']}/{summary['fixture_count']} fixture(s), "
			f"{len(report['constructs_covered'])}/{len(report['constructs_required'])} construct(s)"
		)
		for gap in report["blocking_gaps"]:
			click.echo(f"  error: {gap['id']}: {'; '.join(gap['errors'])}")

	if not report["ok"]:
		raise click.exceptions.Exit(1)
