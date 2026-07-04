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
@click.option(
	"--refresh",
	"refresh_outputs",
	is_flag=True,
	help="Alias for --refresh-outputs",
)
@click.option(
	"--refresh-outputs",
	"refresh_outputs",
	is_flag=True,
	help="Rewrite each numbered example output directory from the current compiler before auditing",
)
@click.option(
	"--update",
	is_flag=True,
	help="Recompile all numbered examples and regenerate their output/ directories, then report results",
)
def baseline(examples_dir: Path, as_json: bool, refresh_outputs: bool, update: bool) -> None:
	"""Run the compiler bed-down gate over numbered APG examples."""
	if not examples_dir.exists():
		raise click.ClickException(f"Examples directory not found: {examples_dir}")
	if not examples_dir.is_dir():
		raise click.ClickException(f"APG baseline expects a directory: {examples_dir}")

	if update:
		_run_update(examples_dir, as_json)
		return

	report = build_compiler_baseline_report(examples_dir, refresh_outputs=refresh_outputs)
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


def _run_update(examples_dir: Path, as_json: bool) -> None:
	"""Recompile every numbered example and regenerate its output/ directory."""
	from compiler.baseline import _refresh_output_directory
	from compiler.compiler import APGCompiler

	sources = sorted(examples_dir.glob("[0-9][0-9]_*/main.apg"))
	if not sources:
		raise click.ClickException(f"No numbered examples found in {examples_dir}")

	click.echo(f"Updating {len(sources)} example(s)...")

	results: list[dict] = []
	updated = 0
	for source in sources:
		compiler = APGCompiler()
		try:
			result = compiler.compile_file(source, target_language="python")
		except Exception as exc:
			click.echo(f"  {source.parent.name}: FAILED ({exc})")
			results.append({"name": source.parent.name, "ok": False, "error": str(exc)})
			continue

		if result.success:
			output_dir = source.parent / "output"
			# Clear stale files first
			if output_dir.exists():
				import shutil
				shutil.rmtree(output_dir)
			_refresh_output_directory(output_dir, result.generated_files)
			click.echo(f"  {source.parent.name}: OK (regenerated)")
			updated += 1
			results.append({"name": source.parent.name, "ok": True})
		else:
			errs = "; ".join(str(e) for e in result.errors)
			click.echo(f"  {source.parent.name}: FAILED ({errs})")
			results.append({"name": source.parent.name, "ok": False, "error": errs})

	click.echo(f"Updated {updated}/{len(sources)} example(s)")

	if as_json:
		click.echo(json.dumps({"updated": updated, "total": len(sources), "results": results}, indent=2))


if __name__ == "__main__":
	baseline()
