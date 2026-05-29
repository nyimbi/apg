#!/usr/bin/env python3
"""APG release evidence command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.release import build_release_report


@click.command(name="release")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.option("--target", default="python", help="Release target to verify")
@click.option("--catalog", type=click.Path(path_type=Path), default=None, help="Capability contract root or local apg.capability-catalog.v1 file")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.release-report.v1 JSON")
def release(source_file: Path, target: str, catalog: Path | None, as_json: bool) -> None:
	"""Compile source and emit generated application release evidence."""
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG release expects a file: {source_file}")
	if catalog is not None and not catalog.exists():
		raise click.ClickException(f"Capability catalog not found: {catalog}")

	report = build_release_report(source_file, target=target, catalog=catalog)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		generated = report.get("generated", {})
		evidence = report.get("evidence", {})
		self_test = evidence.get("self_test", {})
		click.echo(
			f"APG release {status}: {source_file}, "
			f"{generated.get('file_count', 0)} artifact(s), "
			f"self-test={self_test.get('status', 'unknown')}"
		)
		for error in report.get("errors", []):
			click.echo(f"  error: {error}")
		for warning in report.get("warnings", []):
			click.echo(f"  warning: {warning}")

	if not report["ok"]:
		raise click.exceptions.Exit(1)
