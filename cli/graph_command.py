#!/usr/bin/env python3
"""APG graph command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.graphs import build_graph, render_dot, render_mermaid


@click.command(name="graph")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.option(
	"--kind",
	default="er",
	type=click.Choice(
		["er", "lookup", "workflow", "handler", "capability", "security", "agent", "deployment", "package"],
		case_sensitive=False,
	),
	help="Graph kind to emit",
)
@click.option(
	"--format",
	"output_format",
	default="json",
	type=click.Choice(["json", "mermaid", "dot"], case_sensitive=False),
	help="Graph output format",
)
def graph(source_file: Path, kind: str, output_format: str) -> None:
	"""Emit an APG graph without generating application code."""
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG graph expects a file: {source_file}")

	try:
		apg_graph = build_graph(source_file, kind)
	except ValueError as error:
		raise click.ClickException(str(error)) from error

	normalized_format = output_format.lower()
	if normalized_format == "json":
		click.echo(json.dumps(apg_graph.to_dict(), indent=2, sort_keys=True))
	elif normalized_format == "mermaid":
		click.echo(render_mermaid(apg_graph), nl=False)
	else:
		click.echo(render_dot(apg_graph), nl=False)
