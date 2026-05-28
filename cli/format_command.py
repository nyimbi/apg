#!/usr/bin/env python3
"""APG format command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.formatter import audit_formatter_fixtures, format_apg_source


def format_file(path: Path, write: bool = False, include_text: bool = True) -> dict[str, object]:
	source = path.read_text(encoding="utf-8")
	result = format_apg_source(source)
	if write and result.changed:
		path.write_text(result.text, encoding="utf-8")
	payload = result.to_dict(include_text=include_text)
	payload["file"] = str(path)
	payload["written"] = bool(write and result.changed)
	return payload


@click.command(name="format")
@click.argument("source_file", required=False, type=click.Path(path_type=Path))
@click.option("--check", is_flag=True, help="Exit 1 when formatting changes are needed")
@click.option("--write", "write_changes", is_flag=True, help="Write formatted APG source in place")
@click.option("--audit-fixtures", is_flag=True, help="Audit checked-in formatter fixtures")
@click.option("--fixture-catalog", type=click.Path(path_type=Path), help="Formatter fixture catalog to audit")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON")
def format_cmd(
	source_file: Path | None,
	check: bool,
	write_changes: bool,
	audit_fixtures: bool,
	fixture_catalog: Path | None,
	as_json: bool,
) -> None:
	"""Format one APG source file deterministically."""
	if check and write_changes:
		raise click.ClickException("--check and --write cannot be combined")
	if audit_fixtures:
		if source_file is not None or check or write_changes:
			raise click.ClickException("--audit-fixtures cannot be combined with a source file, --check, or --write")
		payload = audit_formatter_fixtures(fixture_catalog)
		if as_json:
			click.echo(json.dumps(payload, indent=2, sort_keys=True))
		else:
			summary = payload["summary"]
			status = "ok" if payload["ok"] else "failed"
			click.echo(
				f"Formatter fixtures {status}: "
				f"{summary['passing_fixture_count']}/{summary['fixture_count']} passing, "
				f"{summary['blocking_gap_count']} blocking gaps"
			)
		if not payload["ok"]:
			raise click.exceptions.Exit(1)
		return
	if source_file is None:
		raise click.ClickException("APG format requires a source file unless --audit-fixtures is used")
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG format expects a file: {source_file}")

	payload = format_file(
		source_file,
		write=write_changes,
		include_text=as_json or not check,
	)
	if as_json:
		click.echo(json.dumps(payload, indent=2, sort_keys=True))
	elif check:
		status = "would change" if payload["changed"] else "already formatted"
		click.echo(f"{source_file}: {status}")
	elif write_changes:
		status = "formatted" if payload["written"] else "already formatted"
		click.echo(f"{source_file}: {status}")
	else:
		click.echo(payload["text"], nl=False)

	if check and payload["changed"]:
		raise click.exceptions.Exit(1)
