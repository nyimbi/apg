#!/usr/bin/env python3
"""APG schema DDL generation command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.schema import generate_schema


@click.command(name="schema")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.option(
	"--dialect",
	default="postgresql",
	show_default=True,
	type=click.Choice(["postgresql", "sqlite", "mysql"]),
	help="SQL dialect for generated DDL",
)
@click.option("--json", "as_json", is_flag=True, help="Emit apg.schema-report.v1 JSON")
@click.option(
	"--out",
	type=click.Path(path_type=Path),
	default=None,
	help="Write DDL to file instead of stdout",
)
def schema(source_file: Path, dialect: str, as_json: bool, out: Path | None) -> None:
	"""Generate SQL DDL from APG table declarations."""
	if not source_file.exists():
		raise click.ClickException(f"File not found: {source_file}")
	report = generate_schema(source_file, dialect)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		if out:
			out.write_text(report["ddl"], encoding="utf-8")
			click.echo(f"Wrote {report['table_count']} table(s) to {out}")
		else:
			click.echo(report["ddl"])
	if not report["ok"]:
		raise click.exceptions.Exit(1)
