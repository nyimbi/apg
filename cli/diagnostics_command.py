#!/usr/bin/env python3
"""APG diagnostic registry command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.diagnostics import audit_diagnostic_fixtures, diagnostic_registry


@click.command(name="diagnostics")
@click.option("--audit-fixtures", is_flag=True, help="Audit diagnostic fixture coverage")
@click.option("--fixtures", type=click.Path(path_type=Path), default=None, help="Diagnostic fixture directory")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON")
def diagnostics(audit_fixtures: bool, fixtures: Path | None, as_json: bool) -> None:
	"""Inspect or audit APG diagnostic registry coverage."""
	if audit_fixtures:
		report = audit_diagnostic_fixtures(fixtures)
		if as_json:
			click.echo(json.dumps(report, indent=2, sort_keys=True))
		else:
			status = "OK" if report["ok"] else "FAILED"
			summary = report["summary"]
			click.echo(
				f"APG diagnostics {status}: "
				f"{summary['fixture_count']}/{summary['registry_count']} fixture(s), "
				f"{summary['missing_fixture_count']} missing, "
				f"{summary['unknown_fixture_count']} unknown"
			)
			for diagnostic in report["diagnostics"]:
				click.echo(f"  {diagnostic['code']} {diagnostic['severity']}: {diagnostic['message']}")
		if not report["ok"]:
			raise click.exceptions.Exit(1)
		return

	registry = diagnostic_registry()
	if as_json:
		click.echo(json.dumps({
			"format": "apg.diagnostic-registry.v1",
			"ok": True,
			"registry": registry,
			"summary": {"registry_count": len(registry)},
		}, indent=2, sort_keys=True))
	else:
		click.echo(f"APG diagnostics registry: {len(registry)} code(s)")
		for code, entry in registry.items():
			click.echo(f"  {code} {entry['severity']}: {entry['title']}")
