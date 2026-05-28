#!/usr/bin/env python3
"""APG graph command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.graphs import SUPPORTED_GRAPH_KINDS, build_graph, build_graph_suite, render_dot, render_mermaid


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


@click.command(name="graph-suite")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit apg.graph-suite-report.v1 JSON")
def graph_suite(source_file: Path, as_json: bool) -> None:
	"""Emit every supported APG graph kind and rendering."""
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG graph-suite expects a file: {source_file}")

	try:
		report = build_graph_suite(source_file)
	except ValueError as error:
		raise click.ClickException(str(error)) from error

	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
		return

	click.echo(f"APG graph-suite OK: {source_file}")
	for kind in SUPPORTED_GRAPH_KINDS:
		counts = report["summary"][kind]
		click.echo(f"  {kind}: {counts['nodes']} node(s), {counts['edges']} edge(s)")
