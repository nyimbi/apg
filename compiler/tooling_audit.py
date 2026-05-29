"""Aggregate fixture audit for APG tooling contracts."""

from __future__ import annotations

from typing import Any, Callable

from compiler.diagnostics import audit_diagnostic_fixtures
from compiler.drift import audit_drift_fixtures
from compiler.evidence_bundle import audit_release_evidence_fixtures
from compiler.formatter import audit_formatter_fixtures
from compiler.graphs import audit_graph_fixtures
from compiler.ide_integration import audit_vscode_extension
from compiler.linting import audit_lint_fixtures
from compiler.migrations import audit_migration_fixtures
from compiler.nl_plan import audit_nl_plan_fixtures
from compiler.parser_golden import audit_parser_golden
from compiler.semantic_model import audit_semantic_model_fixtures
from compiler.studio import build_studio_edit_plan, build_studio_snapshot
from language_server.semantic_service import audit_language_server_fixtures


TOOLING_FIXTURE_AUDIT_FORMAT = "apg.tooling-fixture-audit.v1"
CLI_SURFACE_AUDIT_FORMAT = "apg.cli-surface-audit.v1"
STUDIO_SURFACE_AUDIT_FORMAT = "apg.studio-surface-audit.v1"

REQUIRED_TOP_LEVEL_COMMANDS = [
	"baseline",
	"capabilities",
	"compile",
	"create",
	"deployment",
	"diagnostics",
	"doctor",
	"drift",
	"evidence",
	"explain",
	"format",
	"graph",
	"graph-suite",
	"ide",
	"init",
	"language-server",
	"lint",
	"migrate-plan",
	"model",
	"nl-plan",
	"package",
	"package-verify",
	"parser-golden",
	"release",
	"run",
	"studio",
	"tooling",
	"validate",
	"version",
]

REQUIRED_COMMAND_GROUPS = {
	"capabilities": ["contracts", "evaluate-rules", "inspect", "list", "publish-plan", "scaffold", "validate-contracts"],
	"deployment": ["verify"],
	"ide": ["audit"],
	"studio": ["plan-edit", "snapshot"],
	"tooling": ["audit"],
}

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
		("lint", "apg.lint-fixture-audit.v1", audit_lint_fixtures),
		("formatter", "apg.formatter-audit.v1", audit_formatter_fixtures),
		("drift", "apg.drift-audit.v1", audit_drift_fixtures),
		("semantic_model", "apg.semantic-model-fixture-audit.v1", audit_semantic_model_fixtures),
		("graph", "apg.graph-fixture-audit.v1", audit_graph_fixtures),
		("language_server", "apg.language-server-fixture-audit.v1", audit_language_server_fixtures),
		("nl_plan", "apg.nl-plan-fixture-audit.v1", audit_nl_plan_fixtures),
		("migration", "apg.migration-fixture-audit.v1", audit_migration_fixtures),
		("release_evidence", "apg.release-evidence-fixture-audit.v1", audit_release_evidence_fixtures),
		("cli_surface", CLI_SURFACE_AUDIT_FORMAT, audit_cli_surface_contracts),
		("ide_integration", "apg.ide-audit.v1", audit_vscode_extension),
		("studio_designer", STUDIO_SURFACE_AUDIT_FORMAT, audit_studio_designer_surface),
	]


def audit_cli_surface_contracts() -> dict[str, Any]:
	"""Verify that APG's documented CLI tooling surface is registered."""
	from click.testing import CliRunner

	from cli.main import cli

	runner = CliRunner()
	top_level_help = runner.invoke(cli, ["--help"])
	registered_commands = sorted(cli.commands)
	command_checks: list[dict[str, Any]] = []
	errors: list[str] = []

	if top_level_help.exit_code != 0:
		errors.append(f"apg --help failed with exit code {top_level_help.exit_code}")

	missing_commands = sorted(set(REQUIRED_TOP_LEVEL_COMMANDS).difference(registered_commands))
	if missing_commands:
		errors.append(f"missing top-level commands: {', '.join(missing_commands)}")

	for command, subcommands in REQUIRED_COMMAND_GROUPS.items():
		result = runner.invoke(cli, [command, "--help"])
		output = result.output or ""
		missing_subcommands = [
			subcommand
			for subcommand in subcommands
			if subcommand not in output
		]
		ok = result.exit_code == 0 and not missing_subcommands
		if not ok:
			if result.exit_code != 0:
				errors.append(f"apg {command} --help failed with exit code {result.exit_code}")
			if missing_subcommands:
				errors.append(f"apg {command} missing subcommands: {', '.join(missing_subcommands)}")
		command_checks.append({
			"command": command,
			"ok": ok,
			"missing_subcommands": missing_subcommands,
			"exit_code": result.exit_code,
		})

	for forbidden in ("flask-appbuilder", "flask_appbuilder", "--target django", "--target flask"):
		if forbidden in top_level_help.output:
			errors.append(f"top-level CLI help advertises forbidden target fragment: {forbidden}")

	return {
		"format": CLI_SURFACE_AUDIT_FORMAT,
		"ok": not errors,
		"registered_commands": registered_commands,
		"required_commands": REQUIRED_TOP_LEVEL_COMMANDS,
		"missing_commands": missing_commands,
		"command_groups": command_checks,
		"summary": {
			"registered_command_count": len(registered_commands),
			"required_command_count": len(REQUIRED_TOP_LEVEL_COMMANDS),
			"missing_command_count": len(missing_commands),
			"command_group_count": len(command_checks),
			"passing_command_group_count": sum(1 for check in command_checks if check["ok"]),
			"blocking_gap_count": len(errors),
		},
		"errors": errors,
		"blocking_gaps": [
			{"surface": "cli", "error": error}
			for error in errors
		],
	}


def audit_studio_designer_surface() -> dict[str, Any]:
	"""Verify Studio snapshot and visual-edit planning on a real APG source."""
	from pathlib import Path

	source = Path(__file__).resolve().parents[1] / "examples" / "11_screen_composition_relationships" / "main.apg"
	errors: list[str] = []
	checks: list[dict[str, Any]] = []

	try:
		snapshot = build_studio_snapshot(source)
	except Exception as error:
		snapshot = {}
		errors.append(f"studio snapshot failed: {error}")

	_panels = snapshot.get("panels", {}) if isinstance(snapshot, dict) else {}
	snapshot_ok = (
		bool(snapshot.get("ok"))
		and snapshot.get("format") == "apg.studio-snapshot.v1"
		and all(
			panel in _panels
			for panel in [
				"dsl_editor",
				"component_palette",
				"database_designer",
				"form_designer",
				"workflow_designer",
				"capability_composition_designer",
				"package_deployment_designer",
				"graph_explain_panel",
			]
		)
	)
	if not snapshot_ok:
		errors.append("studio snapshot did not expose all expected designer panels")
	checks.append({
		"name": "snapshot",
		"ok": snapshot_ok,
		"format": snapshot.get("format"),
		"panel_count": len(_panels),
	})

	edit = {
		"operation": "add_screen",
		"name": "ExceptionReview",
		"route": "/ops/exceptions",
		"title": "Exception Review",
	}
	try:
		edit_plan = build_studio_edit_plan(source, edit, write=False)
	except Exception as error:
		edit_plan = {}
		errors.append(f"studio edit planning failed: {error}")

	edit_ok = (
		bool(edit_plan.get("ok"))
		and edit_plan.get("format") == "apg.studio-edit-plan.v1"
		and edit_plan.get("changed") is True
		and edit_plan.get("written") is False
		and "screen ExceptionReview" in str(edit_plan.get("new_source", ""))
	)
	if not edit_ok:
		errors.append("studio edit planning did not produce a reviewable dry-run APG patch")
	checks.append({
		"name": "plan_edit",
		"ok": edit_ok,
		"format": edit_plan.get("format"),
		"changed": edit_plan.get("changed"),
		"written": edit_plan.get("written"),
	})

	return {
		"format": STUDIO_SURFACE_AUDIT_FORMAT,
		"ok": not errors,
		"source": str(source),
		"checks": checks,
		"summary": {
			"check_count": len(checks),
			"passing_check_count": sum(1 for check in checks if check["ok"]),
			"blocking_gap_count": len(errors),
		},
		"errors": errors,
		"blocking_gaps": [
			{"surface": "studio", "error": error}
			for error in errors
		],
	}


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
