#!/usr/bin/env python3
"""APG IDE integration commands."""

from __future__ import annotations

import json

import click

from compiler.ide_integration import audit_vscode_extension


@click.group()
def ide() -> None:
	"""Inspect APG IDE integration contracts."""


@ide.command(name="audit")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.ide-audit.v1 JSON")
def audit(as_json: bool) -> None:
	"""Audit the checked-in APG IDE integration surfaces."""
	report = audit_vscode_extension()
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(
			f"APG IDE audit {status}: {report['surface']}, "
			f"{report['summary']['passing']}/{report['summary']['check_count']} checks passing"
		)
		for check in report["checks"]:
			prefix = "OK" if check["ok"] else "FAIL"
			click.echo(f"  {prefix} {check['name']}: {check['message']}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)
