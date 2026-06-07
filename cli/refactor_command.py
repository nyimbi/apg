#!/usr/bin/env python3
"""APG source refactoring commands."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group(name="refactor")
def refactor():
	"""Refactor APG source files."""


@refactor.command(name="rename-entity")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.argument("old_name")
@click.argument("new_name")
@click.option("--write", is_flag=True, help="Apply the rename to the source file")
@click.option("--json", "as_json", is_flag=True)
def rename_entity_cmd(source_file: Path, old_name: str, new_name: str, write: bool, as_json: bool) -> None:
	"""Rename an entity everywhere in SOURCE_FILE."""
	from compiler.refactor import rename_entity
	if not source_file.exists():
		raise click.ClickException(f"File not found: {source_file}")
	report = rename_entity(source_file, old_name, new_name, write=write)
	if as_json:
		out = dict(report)
		out.pop("new_source", None)
		click.echo(json.dumps(out, indent=2))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(f"Refactor {status}: {old_name} -> {new_name} ({report.get('occurrences', 0)} occurrences)")
		if report.get("diff"):
			click.echo(report["diff"])
		if not report["ok"] and report.get("errors"):
			for err in report["errors"]:
				click.echo(f"  error: {err}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@refactor.command(name="rename-field")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.argument("entity_name")
@click.argument("old_field")
@click.argument("new_field")
@click.option("--write", is_flag=True)
@click.option("--json", "as_json", is_flag=True)
def rename_field_cmd(source_file: Path, entity_name: str, old_field: str, new_field: str, write: bool, as_json: bool) -> None:
	"""Rename ENTITY_NAME.OLD_FIELD to NEW_FIELD in SOURCE_FILE."""
	from compiler.refactor import rename_field
	if not source_file.exists():
		raise click.ClickException(f"File not found: {source_file}")
	report = rename_field(source_file, entity_name, old_field, new_field, write=write)
	if as_json:
		out = dict(report)
		out.pop("new_source", None)
		click.echo(json.dumps(out, indent=2))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(f"Refactor {status}: {entity_name}.{old_field} -> {new_field}")
		if report.get("diff"):
			click.echo(report["diff"])
		if not report["ok"] and report.get("errors"):
			for err in report["errors"]:
				click.echo(f"  error: {err}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)
