"""Parser golden fixture audit for APG grammar coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .parser import APGParser, APGSyntaxError


DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "parser_golden" / "catalog.json"


def audit_parser_golden(catalog_path: Path | None = None) -> dict[str, Any]:
	"""Run the checked-in parser golden fixture catalog."""
	catalog_file = Path(catalog_path or DEFAULT_CATALOG)
	catalog_root = catalog_file.parent
	catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
	required = list(catalog.get("constructs_required", []))
	covered: set[str] = set()
	fixture_reports: list[dict[str, Any]] = []
	blocking_gaps: list[dict[str, Any]] = []

	for fixture in catalog.get("fixtures", []):
		report = _audit_fixture(catalog_root, fixture)
		fixture_reports.append(report)
		if report["expected_valid"] and report["ok"]:
			covered.update(report["constructs"])
		if not report["ok"]:
			blocking_gaps.append({
				"id": report["id"],
				"path": report["path"],
				"expected_valid": report["expected_valid"],
				"actual_valid": report["actual_valid"],
				"errors": report["errors"],
			})

	missing = sorted(set(required).difference(covered))
	for construct in missing:
		blocking_gaps.append({
			"id": f"missing_construct:{construct}",
			"path": str(catalog_file),
			"expected_valid": True,
			"actual_valid": False,
			"errors": [f"required construct {construct!r} is not covered by a passing valid fixture"],
		})

	return {
		"format": "apg.parser-golden-audit.v1",
		"ok": not blocking_gaps,
		"catalog": str(catalog_file),
		"constructs_required": required,
		"constructs_covered": sorted(covered),
		"missing_constructs": missing,
		"fixtures": fixture_reports,
		"summary": {
			"fixture_count": len(fixture_reports),
			"valid_fixture_count": sum(1 for report in fixture_reports if report["expected_valid"]),
			"invalid_fixture_count": sum(1 for report in fixture_reports if not report["expected_valid"]),
			"passing_fixture_count": sum(1 for report in fixture_reports if report["ok"]),
			"blocking_gap_count": len(blocking_gaps),
		},
		"blocking_gaps": blocking_gaps,
	}


def _audit_fixture(catalog_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
	fixture_id = str(fixture["id"])
	fixture_path = (catalog_root / str(fixture["path"])).resolve()
	expected_valid = bool(fixture.get("valid", True))
	constructs = sorted(str(construct) for construct in fixture.get("constructs", []))
	errors: list[str] = []
	warnings: list[str] = []
	actual_valid = False

	try:
		source = fixture_path.read_text(encoding="utf-8")
		parse_result = APGParser().parse_string(source, str(fixture_path))
		actual_valid = bool(parse_result.get("success"))
		errors = [_format_error(error) for error in parse_result.get("errors", [])]
		warnings = [str(warning) for warning in parse_result.get("warnings", [])]
	except Exception as error:
		errors = [str(error)]

	return {
		"id": fixture_id,
		"path": str(fixture_path),
		"expected_valid": expected_valid,
		"actual_valid": actual_valid,
		"ok": actual_valid is expected_valid,
		"constructs": constructs,
		"errors": errors,
		"warnings": warnings,
	}


def _format_error(error: object) -> str:
	if isinstance(error, APGSyntaxError):
		return f"{error.line}:{error.column}: {error.message}"
	return str(error)
