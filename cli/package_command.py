#!/usr/bin/env python3
"""APG package command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.packager import SUPPORTED_PACKAGE_TARGETS, build_package_report


@click.command(name="package")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.option(
	"--target",
	default="web",
	type=click.Choice(SUPPORTED_PACKAGE_TARGETS, case_sensitive=False),
	help="Packaging profile",
)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path("dist"), help="Package output root")
@click.option("--catalog", type=click.Path(path_type=Path), default=None, help="Capability contract root or local apg.capability-catalog.v1 file")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.package-report.v1 JSON")
def package(source_file: Path, target: str, out_dir: Path, catalog: Path | None, as_json: bool) -> None:
	"""Package generated Python artifacts for an APG profile."""
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG package expects a file: {source_file}")
	if catalog is not None and not catalog.exists():
		raise click.ClickException(f"Capability catalog not found: {catalog}")

	report = build_package_report(source_file, target=target, out_dir=out_dir, catalog=catalog)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(
			f"APG package {status}: {source_file}, "
			f"target={report['target']}, output={report['output_dir']}"
		)
		for warning in report["warnings"]:
			click.echo(f"  warning: {warning}")
		for error in report["errors"]:
			click.echo(f"  error: {error}")

	if not report["ok"]:
		raise click.exceptions.Exit(1)
