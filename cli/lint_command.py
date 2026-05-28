#!/usr/bin/env python3
"""APG lint command.

The linter is intentionally dependency-light: it reuses the existing parser,
AST builder, and semantic analyzer, then emits a stable report shape for humans,
CI, IDEs, and agent workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from compiler.ast_builder import ASTBuilder
from compiler.parser import APGParser, APGSyntaxError
from compiler.semantic_analyzer import SemanticAnalyzer, SemanticError


console = Console()


def _position(line: int | None, column: int | None) -> dict[str, int]:
	return {
		"line": max(0, int(line or 1) - 1),
		"character": max(0, int(column or 0)),
	}


def _diagnostic_range(line: int | None, column: int | None) -> dict[str, dict[str, int]]:
	start = _position(line, column)
	return {
		"start": start,
		"end": {"line": start["line"], "character": start["character"] + 1},
	}


def _diagnostic_from_error(
	error: APGSyntaxError | SemanticError | Exception,
	file_path: Path,
	severity: str,
) -> dict[str, Any]:
	if isinstance(error, APGSyntaxError):
		return {
			"code": "APG0001",
			"title": "Syntax error",
			"severity": severity,
			"message": error.message,
			"file": str(file_path),
			"range": _diagnostic_range(error.line, error.column),
			"related_locations": [],
			"fixes": [],
			"docs_url": "docs/tooling.md#diagnostic-specification",
		}

	if isinstance(error, SemanticError):
		node = error.node
		title = (
			"Semantic warning"
			if error.error_type == "warning"
			else f"{error.error_type.title()} error"
		)
		return {
			"code": "APG0200" if error.error_type == "type" else "APG0100",
			"title": title,
			"severity": severity,
			"message": error.message,
			"file": str(file_path),
			"range": _diagnostic_range(
				getattr(node, "line", None),
				getattr(node, "column", None),
			),
			"related_locations": [],
			"fixes": [],
			"docs_url": "docs/tooling.md#diagnostic-specification",
		}

	return {
		"code": "APG9000",
		"title": "Internal tooling error",
		"severity": "error",
		"message": str(error),
		"file": str(file_path),
		"range": _diagnostic_range(1, 0),
		"related_locations": [],
		"fixes": [],
		"docs_url": "docs/tooling.md#diagnostic-specification",
	}


def _lint_file(file_path: Path, strict: bool = False) -> dict[str, Any]:
	parser = APGParser()
	ast_builder = ASTBuilder()
	analyzer = SemanticAnalyzer()
	diagnostics: list[dict[str, Any]] = []
	semantic_model_available = False

	try:
		parse_result = parser.parse_file(str(file_path))
	except Exception as error:
		diagnostics.append(_diagnostic_from_error(error, file_path, "error"))
		return _file_report(file_path, diagnostics, semantic_model_available, strict)

	for error in parse_result.get("errors", []):
		diagnostics.append(_diagnostic_from_error(error, file_path, "error"))

	if parse_result.get("success"):
		try:
			ast = parse_result.get("ast") or ast_builder.build_ast(
				parse_result["parse_tree"],
				str(file_path),
			)
			if ast is None:
				raise RuntimeError("Failed to build AST")
			semantic_model_available = True
			semantic_result = analyzer.analyze(ast)
			for error in semantic_result.get("errors", []):
				diagnostics.append(_diagnostic_from_error(error, file_path, "error"))
			for warning in semantic_result.get("warnings", []):
				severity = "error" if strict else "warning"
				diagnostics.append(_diagnostic_from_error(warning, file_path, severity))
		except Exception as error:
			diagnostics.append(_diagnostic_from_error(error, file_path, "error"))

	return _file_report(file_path, diagnostics, semantic_model_available, strict)


def _severity_counts(diagnostics: list[dict[str, Any]]) -> dict[str, int]:
	counts = {"error": 0, "warning": 0, "info": 0, "hint": 0}
	for diagnostic in diagnostics:
		severity = diagnostic.get("severity", "error")
		counts[severity] = counts.get(severity, 0) + 1
	return counts


def _file_report(
	file_path: Path,
	diagnostics: list[dict[str, Any]],
	semantic_model_available: bool,
	strict: bool,
) -> dict[str, Any]:
	counts = _severity_counts(diagnostics)
	return {
		"format": "apg.lint-file-report.v1",
		"ok": counts["error"] == 0,
		"file": str(file_path),
		"strict": strict,
		"severity_counts": counts,
		"diagnostics": diagnostics,
		"fixes_available": any(diagnostic.get("fixes") for diagnostic in diagnostics),
		"semantic_model_available": semantic_model_available,
	}


def _source_files(path: Path) -> tuple[str, list[Path]]:
	if path.is_dir():
		return "directory", sorted(
			file_path
			for file_path in path.rglob("*.apg")
			if file_path.is_file()
		)
	return "file", [path]


def lint_path(path: Path, strict: bool = False) -> dict[str, Any]:
	source_mode, files = _source_files(path)
	file_reports = [_lint_file(file_path, strict=strict) for file_path in files]
	diagnostics = [
		diagnostic
		for file_report in file_reports
		for diagnostic in file_report["diagnostics"]
	]
	counts = _severity_counts(diagnostics)
	return {
		"format": "apg.lint-report.v1",
		"ok": bool(files) and counts["error"] == 0,
		"source_mode": source_mode,
		"strict": strict,
		"files": [str(file_path) for file_path in files],
		"severity_counts": counts,
		"diagnostics": diagnostics,
		"fixes_available": any(file_report["fixes_available"] for file_report in file_reports),
		"semantic_model_available": all(
			file_report["semantic_model_available"] for file_report in file_reports
		) if file_reports else False,
		"file_reports": file_reports,
	}


def _print_text_report(report: dict[str, Any]) -> None:
	if not report["files"]:
		console.print("[red]No APG source files found[/red]")
		return

	status = "[green]OK[/green]" if report["ok"] else "[red]FAILED[/red]"
	counts = report["severity_counts"]
	console.print(
		f"APG lint {status}: {len(report['files'])} file(s), "
		f"{counts['error']} error(s), {counts['warning']} warning(s)"
	)
	for diagnostic in report["diagnostics"]:
		start = diagnostic["range"]["start"]
		console.print(
			f"  {diagnostic['file']}:{start['line'] + 1}:{start['character']}: "
			f"{diagnostic['code']} {diagnostic['severity']}: {diagnostic['message']}"
		)


@click.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit apg.lint-report.v1 JSON")
@click.option("--strict", is_flag=True, help="Treat warnings as errors")
@click.option(
	"--catalog",
	type=click.Path(path_type=Path),
	default=None,
	help="Reserved capability catalog path for future semantic checks",
)
def lint(path: Path, as_json: bool, strict: bool, catalog: Path | None) -> None:
	"""Lint APG source without writing generated code."""
	if catalog is not None and not catalog.exists():
		raise click.ClickException(f"Capability catalog not found: {catalog}")
	if not path.exists():
		raise click.ClickException(f"APG path not found: {path}")

	report = lint_path(path, strict=strict)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		_print_text_report(report)

	if not report["ok"]:
		raise click.exceptions.Exit(1)
