#!/usr/bin/env python3
"""APG repository hygiene commands."""

from __future__ import annotations

import json

import click

from compiler.repository_hygiene import audit_repository_hygiene


@click.group(name="hygiene")
def hygiene() -> None:
	"""Audit APG repository layout and root documentation/test placement."""


@hygiene.command(name="audit")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.repository-hygiene-audit.v1 JSON")
@click.option(
	"--include-untracked",
	is_flag=True,
	help="Also report untracked local root clutter and misplaced local docs/tests.",
)
def audit(as_json: bool, include_untracked: bool) -> None:
	"""Verify tracked files follow APG repository hygiene rules."""
	report = audit_repository_hygiene(include_untracked=include_untracked)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		summary = report["summary"]
		click.echo(
			f"APG repository hygiene {status}: "
			f"{summary['passing_check_count']}/{summary['check_count']} checks passing, "
			f"{summary['violation_count']} violation(s)"
		)
		if include_untracked:
			click.echo(f"  Local untracked files inspected: {report['untracked_file_count']}")
		for check in report["checks"]:
			prefix = "OK" if check["ok"] else "FAIL"
			click.echo(f"  {prefix} {check['name']}: {check['violation_count']} violation(s)")
	if not report["ok"]:
		raise click.exceptions.Exit(1)
