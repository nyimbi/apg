"""Aggregate fixture audit for APG tooling contracts."""

from __future__ import annotations

from typing import Any, Callable

from compiler.diagnostics import audit_diagnostic_fixtures
from compiler.drift import audit_drift_fixtures
from compiler.evidence_bundle import audit_release_evidence_fixtures
from compiler.formatter import audit_formatter_fixtures
from compiler.graphs import audit_graph_fixtures
from compiler.migrations import audit_migration_fixtures
from compiler.nl_plan import audit_nl_plan_fixtures
from compiler.parser_golden import audit_parser_golden
from compiler.semantic_model import audit_semantic_model_fixtures
from language_server.semantic_service import audit_language_server_fixtures


TOOLING_FIXTURE_AUDIT_FORMAT = "apg.tooling-fixture-audit.v1"

FixtureAudit = tuple[str, str, Callable[[], dict[str, Any]]]


def audit_tooling_fixtures() -> dict[str, Any]:
	"""Run every checked-in APG tooling fixture audit."""
	surfaces: list[dict[str, Any]] = []
	for name, expected_format, audit in _fixture_audits():
		surfaces.append(_run_surface(name, expected_format, audit))

	blocking_gaps = [
		{
			"surface": surface["name"],
			"format": surface["format"],
			"expected_format": surface["expected_format"],
			"format_ok": surface["format_ok"],
			"errors": surface["errors"],
			"blocking_gaps": surface["blocking_gaps"],
		}
		for surface in surfaces
		if not surface["ok"] or not surface["format_ok"]
	]
	passing_surface_count = sum(1 for surface in surfaces if surface["ok"] and surface["format_ok"])
	return {
		"format": TOOLING_FIXTURE_AUDIT_FORMAT,
		"ok": not blocking_gaps,
		"surface_count": len(surfaces),
		"surfaces": surfaces,
		"summary": {
			"surface_count": len(surfaces),
			"passing_surface_count": passing_surface_count,
			"failing_surface_count": len(surfaces) - passing_surface_count,
			"blocking_gap_count": sum(surface["blocking_gap_count"] for surface in surfaces),
			"error_count": sum(surface["error_count"] for surface in surfaces),
		},
		"blocking_gaps": blocking_gaps,
	}


build_tooling_fixture_audit = audit_tooling_fixtures


def _fixture_audits() -> list[FixtureAudit]:
	return [
		("parser_golden", "apg.parser-golden-audit.v1", audit_parser_golden),
		("diagnostics", "apg.diagnostic-audit.v1", audit_diagnostic_fixtures),
		("formatter", "apg.formatter-audit.v1", audit_formatter_fixtures),
		("drift", "apg.drift-audit.v1", audit_drift_fixtures),
		("semantic_model", "apg.semantic-model-fixture-audit.v1", audit_semantic_model_fixtures),
		("graph", "apg.graph-fixture-audit.v1", audit_graph_fixtures),
		("language_server", "apg.language-server-fixture-audit.v1", audit_language_server_fixtures),
		("nl_plan", "apg.nl-plan-fixture-audit.v1", audit_nl_plan_fixtures),
		("migration", "apg.migration-fixture-audit.v1", audit_migration_fixtures),
		("release_evidence", "apg.release-evidence-fixture-audit.v1", audit_release_evidence_fixtures),
	]


def _run_surface(name: str, expected_format: str, audit: Callable[[], dict[str, Any]]) -> dict[str, Any]:
	try:
		report = audit()
	except Exception as error:
		return {
			"name": name,
			"format": "",
			"expected_format": expected_format,
			"format_ok": False,
			"ok": False,
			"summary": {},
			"errors": [str(error)],
			"blocking_gaps": [],
			"blocking_gap_count": 1,
			"error_count": 1,
		}

	actual_format = str(report.get("format") or "")
	errors = _surface_errors(report)
	blocking_gaps = list(report.get("blocking_gaps") or [])
	blocking_gap_count = _blocking_gap_count(report, errors, blocking_gaps)
	if actual_format != expected_format:
		blocking_gap_count += 1
	return {
		"name": name,
		"format": actual_format,
		"expected_format": expected_format,
		"format_ok": actual_format == expected_format,
		"ok": bool(report.get("ok")),
		"summary": report.get("summary", {}),
		"errors": errors,
		"blocking_gaps": blocking_gaps,
		"blocking_gap_count": blocking_gap_count,
		"error_count": len(errors),
	}


def _surface_errors(report: dict[str, Any]) -> list[Any]:
	errors = list(report.get("errors") or [])
	errors.extend(
		diagnostic
		for diagnostic in report.get("diagnostics") or []
		if diagnostic.get("severity") == "error"
	)
	return errors


def _blocking_gap_count(
	report: dict[str, Any],
	errors: list[Any],
	blocking_gaps: list[Any],
) -> int:
	summary = report.get("summary") or {}
	if "blocking_gap_count" in summary:
		return int(summary["blocking_gap_count"])
	if blocking_gaps:
		return len(blocking_gaps)
	if "failed" in summary:
		return int(summary["failed"])
	return len(errors)
