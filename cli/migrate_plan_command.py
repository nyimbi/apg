#!/usr/bin/env python3
"""APG migration planning command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.migrations import SUPPORTED_MIGRATION_BACKENDS, audit_migration_fixtures, build_migration_plan, _generate_apg_patch


def _parse_rename_hint(_ctx: click.Context, _param: click.Parameter, values: tuple[str, ...]) -> dict[str, str]:
	hints: dict[str, str] = {}
	for value in values:
		if "=" not in value:
			raise click.BadParameter("rename hints must use OLD=NEW")
		old, new = value.split("=", 1)
		old = old.strip()
		new = new.strip()
		if not old or not new:
			raise click.BadParameter("rename hints must include both OLD and NEW")
		hints[old] = new
	return hints


@click.command(name="migrate-plan")
@click.argument("previous", required=False, type=click.Path(path_type=Path))
@click.argument("current", required=False, type=click.Path(path_type=Path))
@click.option("--audit-fixtures", is_flag=True, help="Audit checked-in migration planner fixtures")
@click.option("--fixtures", type=click.Path(path_type=Path), default=None, help="Migration fixture catalog path")
@click.option(
	"--backend",
	default="postgresql",
	type=click.Choice(SUPPORTED_MIGRATION_BACKENDS, case_sensitive=False),
	help="Migration backend profile",
)
@click.option(
	"--rename-hint",
	multiple=True,
	callback=_parse_rename_hint,
	help="Confirm a rename candidate as OLD=NEW, for example table.Customer=Client",
)
@click.option("--json", "as_json", is_flag=True, help="Emit apg.migration-plan.v1 JSON")
@click.option("--patch", "as_patch", is_flag=True, help="Emit an APG DSL patch fragment for additive changes")
def migrate_plan(
	previous: Path | None,
	current: Path | None,
	audit_fixtures: bool,
	fixtures: Path | None,
	backend: str,
	rename_hint: dict[str, str],
	as_json: bool,
	as_patch: bool,
) -> None:
	"""Compare previous/current APG sources or semantic-model JSON files."""
	if audit_fixtures:
		if previous is not None or current is not None or rename_hint:
			raise click.ClickException("--audit-fixtures cannot be combined with migration inputs or --rename-hint")
		report = audit_migration_fixtures(fixtures)
		if as_json:
			click.echo(json.dumps(report, indent=2, sort_keys=True))
		else:
			summary = report["summary"]
			status = "OK" if report["ok"] else "FAILED"
			click.echo(
				f"APG migration fixtures {status}: "
				f"{summary['passing_fixture_count']}/{summary['fixture_count']} passing, "
				f"{summary['blocking_gap_count']} blocking gaps"
			)
		if not report["ok"]:
			raise click.exceptions.Exit(1)
		return
	if previous is None or current is None:
		raise click.ClickException("Specify previous/current migration inputs or use --audit-fixtures")
	for path in (previous, current):
		if not path.exists():
			raise click.ClickException(f"Migration input not found: {path}")
		if not path.is_file():
			raise click.ClickException(f"Migration input must be a file: {path}")

	from compiler.migrations import _load_model
	report = build_migration_plan(previous, current, backend=backend, rename_hints=rename_hint)

	if as_patch:
		old_model = _load_model(previous)
		new_model = _load_model(current)
		patch = _generate_apg_patch(old_model, new_model)
		click.echo(patch)
		return

	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "REVIEW"
		click.echo(
			f"APG migration plan {status}: {previous} -> {current}, "
			f"backend={report['backend']}, changes={len(report['changes'])}, "
			f"destructive={report['destructive']}"
		)
		for diagnostic in report["diagnostics"]:
			click.echo(f"  {diagnostic['code']} {diagnostic['severity']}: {diagnostic['message']}")

	if not report["ok"]:
		raise click.exceptions.Exit(1)
