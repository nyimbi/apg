#!/usr/bin/env python3
"""APG explain command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.explain import build_explain_report


@click.command(name="explain")
@click.argument("source_file", type=click.Path(path_type=Path))
@click.option("--symbol", default=None, help="Explain a semantic-model symbol id or name")
@click.option("--diagnostic", default=None, help="Explain a diagnostic code")
@click.option("--handler", default=None, help="Explain a UI handler or capability screen event")
@click.option("--all-capabilities", "all_capabilities", is_flag=True, help="Explain all capabilities in the file")
@click.option("--all-workflows", "all_workflows", is_flag=True, help="Explain all workflows/flows in the file")
@click.option("--all-rules", "all_rules", default=None, metavar="CAPABILITY_NAME",
              help="Explain all rules in the named capability")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.explain-report.v1 JSON")
def explain(
	source_file: Path,
	symbol: str | None,
	diagnostic: str | None,
	handler: str | None,
	all_capabilities: bool,
	all_workflows: bool,
	all_rules: str | None,
	as_json: bool,
) -> None:
	"""Explain symbols, diagnostics, handlers, and bulk model elements from the APG semantic model.

	Bulk modes (mutually exclusive with each other and with the singular options):

	\b
	  --all-capabilities        Explain every capability defined in SOURCE_FILE.
	  --all-workflows           Explain every workflow/flow defined in SOURCE_FILE.
	  --all-rules CAPABILITY    Explain every rule inside the named capability.
	"""
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG explain expects a file: {source_file}")

	report = build_explain_report(
		source_file,
		symbol=symbol,
		diagnostic=diagnostic,
		handler=handler,
		all_capabilities=all_capabilities,
		all_workflows=all_workflows,
		all_rules=all_rules,
	)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(f"APG explain {status}: {source_file}")
		for explanation in report["explanations"]:
			click.echo(f"  {explanation['summary']}")
		for error in report["errors"]:
			click.echo(f"  error: {error}")

	if not report["ok"]:
		raise click.exceptions.Exit(1)
