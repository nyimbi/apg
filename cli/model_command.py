#!/usr/bin/env python3
"""APG semantic model command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.semantic_model import build_semantic_model


@click.command(name="model")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit apg.semantic-model.v1 JSON")
def model(source_file: Path, as_json: bool) -> None:
	"""Emit the normalized semantic model for APG source."""
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG model expects a file: {source_file}")

	report = build_semantic_model(source_file)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(
			f"APG model {status}: {source_file}, "
			f"{len(report['symbols'])} symbol(s), "
			f"{len(report['tables'])} table(s), "
			f"{len(report['agents'])} agent(s), "
			f"{len(report['capabilities'])} capability(ies)"
		)

	if not report["ok"]:
		raise click.exceptions.Exit(1)
