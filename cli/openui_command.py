#!/usr/bin/env python3
"""APG OpenUI library and prompt command.

Emits the OpenUI ``LibrarySpec`` and system prompt derived from an APG source
file, so a generated app can serve a model-facing screen contract without a
separate build step.

See ``docs/research/openui-generative-ui/README.md`` for the design.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.openui_lang import Library, Program
from compiler.openui_lang.render import render_document, render_html
from compiler.openui_library import build_library, validate_library
from compiler.openui_prompt import generate_system_prompt, prompt_report
from compiler.semantic_model import build_semantic_model


@click.command(name="openui")
@click.argument("source_file", required=False, type=click.Path(path_type=Path))
@click.option("--library", "emit_library", is_flag=True, help="Emit the OpenUI LibrarySpec JSON")
@click.option("--prompt", "emit_prompt", is_flag=True, help="Emit the system prompt")
@click.option("--entities", default="", help="Comma-separated entity names; default is all, capped by --max-entities")
@click.option("--max-entities", type=int, default=12, show_default=True, help="Cap on entity composites, to bound prompt size")
@click.option("--tools", default="", help="Comma-separated tool names exposed to the model")
@click.option("--report", is_flag=True, help="Emit apg.openui-prompt.v1 conformance facts instead of the prompt")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON for --library or --report")
@click.option("--render", "program_file", type=click.Path(path_type=Path), help="Render an OpenUI Lang program to HTML")
@click.option("--fragment", is_flag=True, help="With --render, emit the bare fragment instead of a full page")
@click.option("--state", "state_json", default="", help="With --render, initial state as a JSON object")
def openui(
	source_file: Path | None,
	emit_library: bool,
	emit_prompt: bool,
	entities: str,
	max_entities: int,
	tools: str,
	report: bool,
	as_json: bool,
	program_file: Path | None,
	fragment: bool,
	state_json: str,
) -> None:
	"""Derive an OpenUI component library and system prompt from APG source.

	With ``--render PROGRAM`` the command instead parses that OpenUI Lang
	program against the library and prints the HTML it describes.
	"""
	if source_file is None:
		raise click.ClickException("Specify an APG source file")
	if not source_file.exists():
		raise click.ClickException(f"Source file not found: {source_file}")

	model = build_semantic_model(source_file)
	selected = [name.strip() for name in entities.split(",") if name.strip()] or None
	tool_names = [name.strip() for name in tools.split(",") if name.strip()]

	spec = build_library(model, entities=selected, max_entities=max_entities)
	errors = validate_library(spec)
	if errors:
		for error in errors:
			click.echo(f"library conformance: {error}", err=True)
		raise click.exceptions.Exit(1)

	if program_file is not None:
		_render_program(program_file, spec.to_spec(), state_json, fragment)
		return

	if not emit_library and not emit_prompt and not report:
		emit_prompt = True

	if emit_library:
		document = spec.to_spec()
		if as_json:
			click.echo(json.dumps(document, indent=2, sort_keys=True))
		else:
			click.echo(f"OpenUI library {document['id']} {document['version']}")
			click.echo(f"  root:       {document['root']}")
			click.echo(f"  components: {len(spec.components)}")
			click.echo(f"  functions:  {len(spec.functions)}")
			click.echo(f"  actions:    {len(spec.actions)}")

	prompt = generate_system_prompt(spec, tools=tool_names or None)

	if report:
		facts = prompt_report(spec, prompt)
		if as_json:
			click.echo(json.dumps(facts, indent=2, sort_keys=True))
		else:
			click.echo(f"OpenUI prompt {facts['format']}: {facts['components']} components, {facts['prompt_chars']} chars")
			if facts["missing_signatures"]:
				for missing in facts["missing_signatures"]:
					click.echo(f"  missing: {missing}", err=True)
		if not facts["ok"]:
			raise click.exceptions.Exit(1)

	if emit_prompt and not report:
		click.echo(prompt)


def _render_program(
	program_file: Path, library_document: dict, state_json: str, fragment: bool,
) -> None:
	"""Parse an OpenUI Lang program and print the HTML it describes."""
	if not program_file.exists():
		raise click.ClickException(f"Program file not found: {program_file}")
	state: dict = {}
	if state_json:
		try:
			loaded = json.loads(state_json)
		except json.JSONDecodeError as error:
			raise click.ClickException(f"--state is not valid JSON: {error}") from error
		if not isinstance(loaded, dict):
			raise click.ClickException("--state must be a JSON object")
		state = loaded

	library = Library.from_spec(library_document)
	result = Program.parse(program_file.read_text(encoding="utf-8")).evaluate(library, state=state)

	for error in result.errors:
		click.echo(f"openui: {error.code} in {error.statement_id or 'program'}", err=True)
	if result.unresolved:
		click.echo(f"openui: unresolved {', '.join(sorted(result.unresolved))}", err=True)

	if result.root is None:
		raise click.exceptions.Exit(1)

	if fragment:
		click.echo(render_html(result.root))
	else:
		click.echo(render_document(result.root, title=program_file.stem))
