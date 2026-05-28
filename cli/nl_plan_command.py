#!/usr/bin/env python3
"""APG natural-language plan command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.nl_plan import build_nl_plan


@click.command(name="nl-plan")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.option("--prompt", required=True, help="Natural-language APG edit request")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.nl-plan.v1 JSON")
def nl_plan(source_file: Path, prompt: str, as_json: bool) -> None:
	"""Plan a bounded APG DSL patch without mutating source or generating code."""
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG nl-plan expects a file: {source_file}")

	report = build_nl_plan(source_file, prompt)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(
			f"APG nl-plan {status}: {source_file}, "
			f"intent={report['intent']}, symbols={len(report['affected_symbols'])}"
		)
		if report["dsl_patch"]:
			click.echo(report["dsl_patch"])
		for warning in report["warnings"]:
			click.echo(f"  warning: {warning}")
		for error in report["errors"]:
			click.echo(f"  error: {error}")

	if not report["ok"]:
		raise click.exceptions.Exit(1)
