#!/usr/bin/env python3
"""APG compiler baseline audit command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.baseline import build_compiler_baseline_report


@click.command(name="baseline")
@click.argument(
	"examples_dir",
	required=False,
	type=click.Path(path_type=Path),
	default=Path("examples"),
)
@click.option("--json", "as_json", is_flag=True, help="Emit apg.compiler-baseline-report.v1 JSON")
def baseline(examples_dir: Path, as_json: bool) -> None:
	"""Run the compiler bed-down gate over numbered APG examples."""
	if not examples_dir.exists():
		raise click.ClickException(f"Examples directory not found: {examples_dir}")
	if not examples_dir.is_dir():
		raise click.ClickException(f"APG baseline expects a directory: {examples_dir}")

	report = build_compiler_baseline_report(examples_dir)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		summary = report["summary"]
		click.echo(
			f"APG baseline {status}: {report['example_count']}/{report['expected_examples']} "
			f"example(s), {summary['passed_examples']} passed, {summary['failed_examples']} failed"
		)
		for domain, details in report["domains"].items():
			marker = "ok" if details["ok"] else "missing"
			click.echo(f"  {domain}: {marker} ({len(details['sources'])} example(s))")
		for example in report["examples"]:
			if example["ok"]:
				continue
			click.echo(f"  failed: {example['name']}")
			for error in example.get("errors", []):
				click.echo(f"    error: {error}")

	if not report["ok"]:
		raise click.exceptions.Exit(1)


if __name__ == "__main__":
	baseline()
