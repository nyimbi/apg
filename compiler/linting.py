"""APG lint report builder and fixture audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _format_diagnostic_with_source(diagnostic: dict[str, Any], source_file: Path | None) -> str:
	"""Format a diagnostic with source-line context (Rust/TypeScript style).

	Returns a multi-line string. Falls back to a compact single-line format when
	the source file is unreadable or position info is absent.
	"""
	code = diagnostic.get("code", "APG9999")
	severity = diagnostic.get("severity", "error")
	message = diagnostic.get("message", "")
	file_str = diagnostic.get("file", str(source_file or ""))
	range_info = diagnostic.get("range", {})
	start = range_info.get("start", {})
	line_0 = int(start.get("line", 0))       # 0-based
	col_0 = int(start.get("character", 0))   # 0-based
	line_1 = line_0 + 1                       # 1-based for display

	# Header line: error[CODE]: message
	lines = [f"{severity}[{code}]: {message}"]

	# Arrow line: --> file:line:col
	lines.append(f"  --> {file_str}:{line_1}:{col_0}")

	# Source snippet — only when we have real position info and a readable file
	if line_1 > 0 and source_file is not None:
		try:
			source_lines = source_file.read_text(encoding="utf-8").splitlines()
			if 1 <= line_1 <= len(source_lines):
				src_line = source_lines[line_1 - 1]
				line_prefix = f"{line_1} | "
				blank_prefix = " " * len(str(line_1)) + " | "
				caret_col = max(0, col_0)
				# Determine caret width from end position if available
				end = range_info.get("end", {})
				end_char = int(end.get("character", col_0 + 1)) if end else col_0 + 1
				caret_width = max(1, end_char - col_0)
				caret = " " * caret_col + "^" * caret_width
				lines.append(blank_prefix)
				lines.append(f"{line_prefix}{src_line}")
				lines.append(f"{blank_prefix}{caret}")
		except OSError:
			pass

	return "\n".join(lines)

from capabilities.capability_contract_registry import validate_contract_registry
from compiler.capability_publish import (
	CAPABILITY_CATALOG_FORMAT,
	build_capability_catalog_report,
)
from compiler.parser import APGSyntaxError
from compiler.semantic_analyzer import SemanticError
from compiler.semantic_model import build_semantic_model


LINT_FIXTURE_AUDIT_FORMAT = "apg.lint-fixture-audit.v1"
DEFAULT_LINT_CATALOG = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "lint" / "catalog.json"


def lint_path(path: Path, strict: bool = False, catalog: Path | None = None, collect_all_errors: bool = False) -> dict[str, Any]:
	"""Build an ``apg.lint-report.v1`` report for a file or directory."""
	source_mode, files = _source_files(path)
	file_reports = [_lint_file(file_path, strict=strict, catalog=catalog, collect_all_errors=collect_all_errors) for file_path in files]
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


def audit_lint_fixtures(catalog_path: Path | None = None) -> dict[str, Any]:
	"""Run checked-in linter fixtures against ``apg.lint-report.v1``."""
	catalog_file = Path(catalog_path or DEFAULT_LINT_CATALOG)
	catalog_root = catalog_file.parent
	catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
	required_tags = sorted(str(tag) for tag in catalog.get("tags_required", []))
	covered_tags: set[str] = set()
	fixture_reports: list[dict[str, Any]] = []
	blocking_gaps: list[dict[str, Any]] = []

	for fixture in catalog.get("fixtures", []):
		report = _audit_lint_fixture(catalog_root, fixture)
		fixture_reports.append(report)
		if report["ok"]:
			covered_tags.update(report["tags"])
		else:
			blocking_gaps.append({
				"id": report["id"],
				"source": report["source"],
				"errors": report["errors"],
			})

	missing_tags = sorted(set(required_tags).difference(covered_tags))
	for tag in missing_tags:
		blocking_gaps.append({
			"id": f"missing_tag:{tag}",
			"source": str(catalog_file),
			"errors": [f"required lint fixture tag {tag!r} is not covered by a passing fixture"],
		})

	return {
		"format": LINT_FIXTURE_AUDIT_FORMAT,
		"ok": not blocking_gaps,
		"fixture_catalog": str(catalog_file),
		"tags_required": required_tags,
		"tags_covered": sorted(covered_tags),
		"missing_tags": missing_tags,
		"fixtures": fixture_reports,
		"summary": {
			"fixture_count": len(fixture_reports),
			"passing_fixture_count": sum(1 for report in fixture_reports if report["ok"]),
			"failing_fixture_count": sum(1 for report in fixture_reports if not report["ok"]),
			"blocking_gap_count": len(blocking_gaps),
		},
		"blocking_gaps": blocking_gaps,
	}


def _lint_file(file_path: Path, strict: bool = False, catalog: Path | None = None, collect_all_errors: bool = False) -> dict[str, Any]:
	diagnostics: list[dict[str, Any]] = []
	semantic_model_available = False
	capability_catalog: dict[str, Any] = _empty_catalog_report(catalog)

	try:
		model = build_semantic_model(file_path, collect_all_errors=collect_all_errors)
		semantic_model_available = model.get("format") == "apg.semantic-model.v1"
		diagnostics.extend(_strict_diagnostics(model.get("diagnostics", []), strict))
		if catalog is not None and semantic_model_available:
			capability_catalog, catalog_diagnostics = _capability_catalog_report(file_path, model, catalog)
			diagnostics.extend(catalog_diagnostics)
	except Exception as error:
		diagnostics.append(_diagnostic_from_error(error, file_path, "error"))

	return _file_report(file_path, diagnostics, semantic_model_available, strict, capability_catalog)


def _audit_lint_fixture(catalog_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
	fixture_id = str(fixture["id"])
	source = (catalog_root / str(fixture["source"])).resolve()
	catalog = (catalog_root / str(fixture["catalog"])).resolve() if fixture.get("catalog") else None
	tags = sorted(str(tag) for tag in fixture.get("tags", []))
	expected_ok = bool(fixture.get("expected_ok", True))
	expected_semantic_model = bool(fixture.get("semantic_model_available", True))
	errors: list[str] = []
	report: dict[str, Any] | None = None

	try:
		report = lint_path(source, strict=bool(fixture.get("strict", False)), catalog=catalog)
	except Exception as error:
		errors.append(str(error))

	if report is None:
		return {
			"id": fixture_id,
			"source": str(source),
			"tags": tags,
			"ok": False,
			"format": "",
			"expected_ok": expected_ok,
			"actual_ok": False,
			"diagnostic_codes": [],
			"errors": errors or ["lint report was not produced"],
		}

	if report.get("format") != "apg.lint-report.v1":
		errors.append(f"expected apg.lint-report.v1, got {report.get('format')}")
	if bool(report.get("ok")) != expected_ok:
		errors.append(f"expected ok={expected_ok}, got ok={report.get('ok')}")
	if bool(report.get("semantic_model_available")) != expected_semantic_model:
		errors.append(
			f"expected semantic_model_available={expected_semantic_model}, got {report.get('semantic_model_available')}"
		)

	diagnostic_codes = [str(diagnostic.get("code")) for diagnostic in report.get("diagnostics", [])]
	for code in fixture.get("diagnostic_codes", []):
		if str(code) not in diagnostic_codes:
			errors.append(f"expected diagnostic {code} was not emitted")

	catalog_expectation = fixture.get("capability_catalog")
	if isinstance(catalog_expectation, dict):
		actual_catalog = report.get("capability_catalog", {})
		for key, expected in catalog_expectation.items():
			if actual_catalog.get(key) != expected:
				errors.append(f"expected capability_catalog.{key}={expected!r}, got {actual_catalog.get(key)!r}")

	return {
		"id": fixture_id,
		"source": str(source),
		"catalog": str(catalog) if catalog is not None else None,
		"tags": tags,
		"ok": not errors,
		"format": report.get("format"),
		"expected_ok": expected_ok,
		"actual_ok": bool(report.get("ok")),
		"diagnostic_codes": diagnostic_codes,
		"errors": errors,
	}


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
		"catalog_kind": None,
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
	if catalog.is_file():
		return _local_capability_catalog_report(file_path, model, catalog)
	return _contract_registry_catalog_report(file_path, model, catalog)


def _contract_registry_catalog_report(
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
		"catalog_kind": "contract_registry",
		"contract_count": validation["contract_count"],
		"declared_capabilities": sorted(declared),
		"matched_capabilities": matched,
		"missing_capabilities": missing,
		"errors": list(validation["errors"]),
	}, diagnostics


def _local_capability_catalog_report(
	file_path: Path,
	model: dict[str, Any],
	catalog: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	report = build_capability_catalog_report(catalog)
	diagnostics: list[dict[str, Any]] = []
	for error in report["errors"]:
		diagnostics.append(_catalog_diagnostic(
			file_path,
			code="APG9000",
			title="Capability catalog error",
			message=f"Capability catalog validation failed: {error}",
			severity="error",
		))

	registry_keys = _local_catalog_keys(report.get("records", []))
	declared = model.get("capabilities", {})
	contracts = model.get("contracts", {})
	matched: list[dict[str, Any]] = []
	missing: list[dict[str, Any]] = []

	if report["ok"]:
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
		"ok": report["ok"] and not missing,
		"catalog": str(catalog),
		"catalog_kind": "local_catalog",
		"catalog_format": CAPABILITY_CATALOG_FORMAT,
		"contract_count": report["capability_count"],
		"declared_capabilities": sorted(declared),
		"matched_capabilities": matched,
		"missing_capabilities": missing,
		"errors": list(report["errors"]),
	}, diagnostics


def _local_catalog_keys(records: list[dict[str, Any]]) -> set[str]:
	keys: set[str] = set()
	for record in records:
		for value in [
			record.get("capability"),
			record.get("package"),
		]:
			if isinstance(value, str) and value:
				keys.add(_normalize_capability_key(value))
		for list_key in ("provides", "requires"):
			values = record.get(list_key, [])
			if isinstance(values, list):
				for item in values:
					keys.add(_normalize_capability_key(str(item)))
	return keys


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


def _aggregate_catalog_reports(file_reports: list[dict[str, Any]], catalog: Path | None) -> dict[str, Any]:
	if catalog is None:
		return _empty_catalog_report(None)
	reports = [file_report["capability_catalog"] for file_report in file_reports]
	return {
		"checked": True,
		"ok": bool(reports) and all(report["ok"] for report in reports),
		"catalog": str(catalog),
		"catalog_kind": next((report.get("catalog_kind") for report in reports if report.get("catalog_kind")), None),
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


def _source_files(path: Path) -> tuple[str, list[Path]]:
	if path.is_dir():
		return "directory", sorted(
			file_path
			for file_path in path.rglob("*.apg")
			if file_path.is_file()
		)
	return "file", [path]


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
