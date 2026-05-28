#!/usr/bin/env python3
"""APG Studio and visual designer commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.studio import build_studio_edit_plan, build_studio_snapshot


@click.group()
def studio() -> None:
	"""Inspect and round-trip APG Studio designer state."""


@studio.command(name="snapshot")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit apg.studio-snapshot.v1 JSON")
def snapshot(source_file: Path, as_json: bool) -> None:
	"""Build APG Studio designer state from an APG file."""
	_require_source_file(source_file)
	report = build_studio_snapshot(source_file)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		summary = report["round_trip"]
		panels = report["panels"]
		click.echo(
			f"APG Studio snapshot {'OK' if report['ok'] else 'FAILED'}: {source_file}, "
			f"{len(panels['database_designer']['tables'])} table(s), "
			f"{len(panels['form_designer']['forms'])} form projection(s), "
			f"{len(panels['capability_composition_designer']['capabilities'])} capability(ies), "
			f"{len(summary['supported_edit_operations'])} edit operation(s)"
		)
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@studio.command(name="plan-edit")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.option("--edit-json", required=True, help="Visual edit operation JSON")
@click.option("--write", is_flag=True, help="Apply the visual edit when valid")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.studio-edit-plan.v1 JSON")
def plan_edit(source_file: Path, edit_json: str, write: bool, as_json: bool) -> None:
	"""Plan a visual designer edit as an APG DSL patch."""
	_require_source_file(source_file)
	try:
		edit = json.loads(edit_json)
	except json.JSONDecodeError as error:
		raise click.ClickException(f"--edit-json is invalid JSON: {error}") from error
	if not isinstance(edit, dict):
		raise click.ClickException("--edit-json must decode to an object")

	report = build_studio_edit_plan(source_file, edit, write=write)
	if not write:
		report.pop("new_source", None)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		action = "written" if report["written"] else "planned"
		click.echo(f"APG Studio edit {status}: {report['operation']}, {action}")
		for error in report["errors"]:
			click.echo(f"  - {error}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


def _require_source_file(source_file: Path) -> None:
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG Studio expects a file: {source_file}")
