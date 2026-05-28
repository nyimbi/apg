#!/usr/bin/env python3
"""APG format command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.formatter import format_apg_source


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
@click.argument("source_file", type=click.Path(path_type=Path))
@click.option("--check", is_flag=True, help="Exit 1 when formatting changes are needed")
@click.option("--write", "write_changes", is_flag=True, help="Write formatted APG source in place")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.format-result.v1 JSON")
def format_cmd(source_file: Path, check: bool, write_changes: bool, as_json: bool) -> None:
	"""Format one APG source file deterministically."""
	if check and write_changes:
		raise click.ClickException("--check and --write cannot be combined")
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
