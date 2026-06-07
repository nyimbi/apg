#!/usr/bin/env python3
"""APG validate command.

Validation is the generator-readiness surface. It builds on the same executable
lint report used by `apg lint`, then adds APG target compatibility metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from cli.lint_command import lint_path
from compiler.linting import _format_diagnostic_with_source


console = Console()


def _default_source_file() -> Path | None:
	if Path("apg.json").exists():
		try:
			config = json.loads(Path("apg.json").read_text(encoding="utf-8"))
			source = config.get("build", {}).get("source_file")
			if source and Path(source).exists():
				return Path(source)
		except json.JSONDecodeError:
			return None

	for candidate in (Path("main.apg"), Path("src/main.apg"), Path("app.apg")):
		if candidate.exists():
			return candidate
	return None


def _target_diagnostics(target: str, files: list[str]) -> list[dict[str, Any]]:
	if (target or "python").lower() == "python":
		return []
	file_name = files[0] if files else ""
	return [
		{
			"code": "APG0802",
			"title": "Unsupported compiler target",
			"severity": "error",
			"message": f"APG compiler target must be 'python', not {target!r}.",
			"file": file_name,
			"range": {
				"start": {"line": 0, "character": 0},
				"end": {"line": 0, "character": 1},
			},
			"related_locations": [],
			"fixes": [{"id": "use_python_target", "title": "Use --target python"}],
			"docs_url": "docs/tooling.md#cli-contracts",
		}
	]


def validate_path(
	path: Path,
	target: str = "python",
	strict: bool = False,
	catalog: Path | None = None,
	collect_all_errors: bool = False,
) -> dict[str, Any]:
	lint_report = lint_path(path, strict=strict, catalog=catalog, collect_all_errors=collect_all_errors)
	target_diagnostics = _target_diagnostics(target, lint_report["files"])
	diagnostics = [*lint_report["diagnostics"], *target_diagnostics]
	counts = dict(lint_report["severity_counts"])
	for diagnostic in target_diagnostics:
		severity = diagnostic.get("severity", "error")
		counts[severity] = counts.get(severity, 0) + 1

	return {
		"format": "apg.validate-report.v1",
		"ok": bool(lint_report["files"]) and counts["error"] == 0,
		"target": target,
		"target_compatibility": {
			"requested": target,
			"supported": ["python"],
			"ok": not target_diagnostics,
		},
		"source_mode": lint_report["source_mode"],
		"strict": strict,
		"files": lint_report["files"],
		"severity_counts": counts,
		"diagnostics": diagnostics,
		"lint": lint_report,
		"generator_ready": bool(lint_report["files"]) and counts["error"] == 0,
	}


def _print_plain(report: dict[str, Any]) -> None:
	status = "OK" if report["ok"] else "FAILED"
	counts = report["severity_counts"]
	click.echo(
		f"APG validate {status}: {len(report['files'])} file(s), "
		f"target={report['target']}, {counts['error']} error(s), "
		f"{counts['warning']} warning(s)"
	)
	for diagnostic in report["diagnostics"]:
		file_path = Path(diagnostic.get("file", "")) if diagnostic.get("file") else None
		if file_path and not file_path.exists():
			file_path = None
		click.echo(_format_diagnostic_with_source(diagnostic, file_path))


def _print_table(report: dict[str, Any]) -> None:
	status = "[green]OK[/green]" if report["ok"] else "[red]FAILED[/red]"
	counts = report["severity_counts"]
	console.print(
		f"APG validate {status}: {len(report['files'])} file(s), "
		f"target={report['target']}, {counts['error']} error(s), "
		f"{counts['warning']} warning(s)"
	)

	if not report["diagnostics"]:
		return

	table = Table(show_header=True, header_style="bold magenta")
	table.add_column("File", style="cyan")
	table.add_column("Code", style="yellow")
	table.add_column("Severity")
	table.add_column("Message")
	for diagnostic in report["diagnostics"]:
		table.add_row(
			diagnostic["file"],
			diagnostic["code"],
			diagnostic["severity"],
			diagnostic["message"],
		)
	console.print(table)


@click.command()
@click.argument("source_file", required=False, type=click.Path(path_type=Path))
@click.option("--target", default="python", help="Compiler target to validate")
@click.option(
	"--format",
	"-f",
	"output_format",
	default="table",
	type=click.Choice(["table", "json", "plain"]),
	help="Output format",
)
@click.option("--json", "as_json", is_flag=True, help="Alias for --format json")
@click.option("--warnings", "-w", is_flag=True, help="Compatibility flag; warnings are always counted")
@click.option("--strict", is_flag=True, help="Treat warnings as errors")
@click.option("--catalog", type=click.Path(path_type=Path), default=None, help="Capability contract root or local apg.capability-catalog.v1 file")
@click.option("--recursive", "-r", is_flag=True, help="Validate all APG files in the current directory")
@click.option("--syntax-only", is_flag=True, help="Compatibility flag; validation still builds the shared lint report")
@click.option("--semantic-only", is_flag=True, help="Compatibility flag; validation still builds the shared lint report")
@click.option("--all-errors", "all_errors", is_flag=True, help="Run all semantic analysis phases and collect every error before stopping")
def validate(
	source_file: Path | None,
	target: str,
	output_format: str,
	as_json: bool,
	warnings: bool,
	strict: bool,
	catalog: Path | None,
	recursive: bool,
	syntax_only: bool,
	semantic_only: bool,
	all_errors: bool,
) -> None:
	"""Validate APG source for lint cleanliness and generator readiness."""
	if syntax_only and semantic_only:
		raise click.ClickException("--syntax-only and --semantic-only cannot be combined")

	if recursive:
		path = Path(".")
	else:
		path = source_file or _default_source_file()
		if path is None:
			raise click.ClickException("No APG source file found. Specify a file or create main.apg")
		if not path.exists():
			raise click.ClickException(f"Source file not found: {path}")
	if catalog is not None and not catalog.exists():
		raise click.ClickException(f"Capability catalog not found: {catalog}")

	report = validate_path(path, target=target, strict=strict, catalog=catalog, collect_all_errors=all_errors)
	if as_json or output_format == "json":
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	elif output_format == "plain":
		_print_plain(report)
	else:
		_print_table(report)

	if not report["ok"]:
		raise click.exceptions.Exit(1)


if __name__ == "__main__":
	validate()
