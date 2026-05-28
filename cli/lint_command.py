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

from capabilities.capability_contract_registry import validate_contract_registry
from compiler.parser import APGSyntaxError
from compiler.semantic_analyzer import SemanticError
from compiler.semantic_model import build_semantic_model


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


def _lint_file(file_path: Path, strict: bool = False, catalog: Path | None = None) -> dict[str, Any]:
	diagnostics: list[dict[str, Any]] = []
	semantic_model_available = False
	capability_catalog: dict[str, Any] = _empty_catalog_report(catalog)

	try:
		model = build_semantic_model(file_path)
		semantic_model_available = model.get("format") == "apg.semantic-model.v1"
		diagnostics.extend(_strict_diagnostics(model.get("diagnostics", []), strict))
		if catalog is not None and semantic_model_available:
			capability_catalog, catalog_diagnostics = _capability_catalog_report(file_path, model, catalog)
			diagnostics.extend(catalog_diagnostics)
	except Exception as error:
		diagnostics.append(_diagnostic_from_error(error, file_path, "error"))

	return _file_report(file_path, diagnostics, semantic_model_available, strict, capability_catalog)


def _strict_diagnostics(diagnostics: list[dict[str, Any]], strict: bool) -> list[dict[str, Any]]:
	if not strict:
		return [dict(diagnostic) for diagnostic in diagnostics]
	strict_diagnostics: list[dict[str, Any]] = []
	for diagnostic in diagnostics:
		updated = dict(diagnostic)
		if updated.get("severity") == "warning":
			updated["severity"] = "error"
		strict_diagnostics.append(updated)
	return strict_diagnostics


def _empty_catalog_report(catalog: Path | None) -> dict[str, Any]:
	return {
		"checked": catalog is not None,
		"ok": catalog is None,
		"catalog": str(catalog) if catalog is not None else None,
		"contract_count": 0,
		"declared_capabilities": [],
		"matched_capabilities": [],
		"missing_capabilities": [],
		"errors": [],
	}


def _capability_catalog_report(
	file_path: Path,
	model: dict[str, Any],
	catalog: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	validation = validate_contract_registry(catalog)
	diagnostics: list[dict[str, Any]] = []
	for error in validation["errors"]:
		diagnostics.append(_catalog_diagnostic(
			file_path,
			code="APG9000",
			title="Capability catalog error",
			message=f"Capability catalog validation failed: {error}",
			severity="error",
		))

	registry_keys = {_normalize_capability_key(capability) for capability in validation["capabilities"]}
	declared = model.get("capabilities", {})
	contracts = model.get("contracts", {})
	matched: list[dict[str, Any]] = []
	missing: list[dict[str, Any]] = []

	if validation["valid"]:
		for name, capability in declared.items():
			candidates = _capability_candidate_keys(name, capability, contracts.get(name, {}))
			matched_key = next((candidate for candidate in candidates if _normalize_capability_key(candidate) in registry_keys), None)
			if matched_key is None:
				missing_item = {"name": name, "candidates": candidates}
				missing.append(missing_item)
				diagnostics.append(_catalog_diagnostic(
					file_path,
					code="APG0901",
					title="Unknown capability include",
					message=(
						f"Capability '{name}' does not resolve in catalog {catalog}; "
						f"tried {', '.join(candidates)}."
					),
					severity="error",
					symbol=model.get("symbols", {}).get(f"capability.{name}"),
				))
			else:
				matched.append({"name": name, "matched_key": matched_key})

	return {
		"checked": True,
		"ok": validation["valid"] and not missing,
		"catalog": str(catalog),
		"contract_count": validation["contract_count"],
		"declared_capabilities": sorted(declared),
		"matched_capabilities": matched,
		"missing_capabilities": missing,
		"errors": list(validation["errors"]),
	}, diagnostics


def _capability_candidate_keys(name: str, capability: dict[str, Any], contract: dict[str, Any]) -> list[str]:
	candidates = [name]
	for value in [contract.get("id"), contract.get("capability"), capability.get("name")]:
		if isinstance(value, str) and value:
			candidates.append(value)
	for key in ["provides", "requires"]:
		for item in capability.get(key, []):
			candidates.append(str(item))
	return list(dict.fromkeys(candidates))


def _normalize_capability_key(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(".", "_")


def _catalog_diagnostic(
	file_path: Path,
	code: str,
	title: str,
	message: str,
	severity: str,
	symbol: dict[str, Any] | None = None,
) -> dict[str, Any]:
	range_payload = symbol.get("range") if symbol else _diagnostic_range(1, 0)
	return {
		"code": code,
		"title": title,
		"severity": severity,
		"message": message,
		"file": str(file_path),
		"range": range_payload,
		"related_locations": [],
		"fixes": [],
		"docs_url": "docs/tooling.md#apg-lint",
	}


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
	capability_catalog: dict[str, Any],
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
		"capability_catalog": capability_catalog,
	}


def _source_files(path: Path) -> tuple[str, list[Path]]:
	if path.is_dir():
		return "directory", sorted(
			file_path
			for file_path in path.rglob("*.apg")
			if file_path.is_file()
		)
	return "file", [path]


def lint_path(path: Path, strict: bool = False, catalog: Path | None = None) -> dict[str, Any]:
	source_mode, files = _source_files(path)
	file_reports = [_lint_file(file_path, strict=strict, catalog=catalog) for file_path in files]
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
		"capability_catalog": _aggregate_catalog_reports(file_reports, catalog),
		"file_reports": file_reports,
	}


def _aggregate_catalog_reports(file_reports: list[dict[str, Any]], catalog: Path | None) -> dict[str, Any]:
	if catalog is None:
		return _empty_catalog_report(None)
	reports = [file_report["capability_catalog"] for file_report in file_reports]
	return {
		"checked": True,
		"ok": bool(reports) and all(report["ok"] for report in reports),
		"catalog": str(catalog),
		"contract_count": max((report["contract_count"] for report in reports), default=0),
		"declared_capabilities": sorted({
			capability
			for report in reports
			for capability in report["declared_capabilities"]
		}),
		"matched_capabilities": [
			match
			for report in reports
			for match in report["matched_capabilities"]
		],
		"missing_capabilities": [
			missing
			for report in reports
			for missing in report["missing_capabilities"]
		],
		"errors": [
			error
			for report in reports
			for error in report["errors"]
		],
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
@click.option("--catalog", type=click.Path(path_type=Path), default=None, help="Capability contract catalog root")
def lint(path: Path, as_json: bool, strict: bool, catalog: Path | None) -> None:
	"""Lint APG source without writing generated code."""
	if catalog is not None and not catalog.exists():
		raise click.ClickException(f"Capability catalog not found: {catalog}")
	if not path.exists():
		raise click.ClickException(f"APG path not found: {path}")

	report = lint_path(path, strict=strict, catalog=catalog)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		_print_text_report(report)

	if not report["ok"]:
		raise click.exceptions.Exit(1)
