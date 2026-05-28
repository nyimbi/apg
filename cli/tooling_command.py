#!/usr/bin/env python3
"""APG tooling audit commands."""

from __future__ import annotations

import json

import click

from compiler.tooling_audit import build_tooling_fixture_audit


@click.group(name="tooling")
def tooling() -> None:
	"""Run aggregate APG tooling contract checks."""


@tooling.command(name="audit")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.tooling-fixture-audit.v1 JSON")
def audit(as_json: bool) -> None:
	"""Run every checked-in tooling fixture audit."""
	report = build_tooling_fixture_audit()
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		summary = report["summary"]
		click.echo(
			f"APG tooling audit {status}: "
			f"{summary['passing_surface_count']}/{summary['surface_count']} surfaces passing, "
			f"{summary['blocking_gap_count']} blocking gaps"
		)
		for surface in report["surfaces"]:
			prefix = "OK" if surface["ok"] and surface["format_ok"] else "FAIL"
			click.echo(f"  {prefix} {surface['name']}: {surface['format']}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)
