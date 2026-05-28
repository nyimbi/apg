#!/usr/bin/env python3
"""APG package verification command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.package_verifier import build_package_verification_report


@click.command(name="package-verify")
@click.argument("package_dir", type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit apg.package-verification-report.v1 JSON")
def package_verify(package_dir: Path, as_json: bool) -> None:
	"""Verify an existing APG package profile directory."""
	report = build_package_verification_report(package_dir)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(f"APG package-verify {status}: {package_dir}, profile={report['profile']}")
		for check_name, passed in report["profile_checks"].items():
			click.echo(f"  {check_name}: {'ok' if passed else 'failed'}")
		for error in report["errors"]:
			click.echo(f"  error: {error}")
		for warning in report["warnings"]:
			click.echo(f"  warning: {warning}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


if __name__ == "__main__":
	package_verify()
