#!/usr/bin/env python3
"""APG deployment verification command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.deployment_verifier import build_deployment_verification_report


@click.group(name="deployment")
def deployment() -> None:
	"""Verify generated APG deployment evidence."""


@deployment.command(name="verify")
@click.argument("path", type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit apg.deployment-verification-report.v1 JSON")
def verify(path: Path, as_json: bool) -> None:
	"""Verify generated app or package deployment evidence."""
	report = build_deployment_verification_report(path)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(f"APG deployment verify {status}: {path}")
		for check_name, passed in report["checks"].items():
			click.echo(f"  {check_name}: {'ok' if passed else 'failed'}")
		for error in report["errors"]:
			click.echo(f"  error: {error}")
		for warning in report["warnings"]:
			click.echo(f"  warning: {warning}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


if __name__ == "__main__":
	deployment()
