"""Aggregate tooling audit contract coverage."""

from __future__ import annotations

from compiler.tooling_audit import (
	audit_docs,
	audit_cli_surface_contracts,
	audit_studio_designer_surface,
	audit_tooling_fixtures,
)


def test_tooling_audit_covers_fixture_cli_ide_and_studio_surfaces():
	report = audit_tooling_fixtures()

	assert report["format"] == "apg.tooling-fixture-audit.v1"
	assert report["ok"] is True
	surfaces = {surface["name"]: surface for surface in report["surfaces"]}
	for surface_name in {
		"parser_golden",
		"diagnostics",
		"lint",
		"formatter",
		"drift",
		"semantic_model",
		"graph",
		"capability_implementation",
		"capability_lifecycle",
		"capability_operability",
		"compiler_baseline",
		"repository_hygiene",
		"doctor",
		"docs",
		"language_server",
		"nl_plan",
		"migration",
		"release_evidence",
		"cli_surface",
		"ide_integration",
		"studio_designer",
	}:
		assert surfaces[surface_name]["ok"] is True
		assert surfaces[surface_name]["format_ok"] is True
	assert surfaces["capability_operability"]["summary"]["inoperable_contract_count"] == 0
	assert surfaces["capability_implementation"]["summary"]["capability_count"] >= 100
	assert surfaces["capability_lifecycle"]["summary"]["complete_lifecycle_count"] >= 100
	assert surfaces["capability_lifecycle"]["summary"]["incomplete_lifecycle_count"] == 0
	assert surfaces["compiler_baseline"]["summary"]["passed_examples"] == 20
	assert surfaces["compiler_baseline"]["summary"]["failed_examples"] == 0
	assert report["summary"]["surface_count"] == 21
	assert report["summary"]["blocking_gap_count"] == 0


def test_cli_surface_audit_tracks_documented_command_groups():
	report = audit_cli_surface_contracts()

	assert report["format"] == "apg.cli-surface-audit.v1"
	assert report["ok"] is True
	assert "compile" in report["registered_commands"]
	assert "docs" in report["registered_commands"]
	assert "hygiene" in report["registered_commands"]
	assert "language-server" in report["registered_commands"]
	assert "package-verify" in report["registered_commands"]
	assert "tooling" in report["registered_commands"]
	assert report["missing_commands"] == []
	groups = {group["command"]: group for group in report["command_groups"]}
	assert groups["capabilities"]["missing_subcommands"] == []
	assert groups["deployment"]["missing_subcommands"] == []
	assert groups["docs"]["missing_subcommands"] == []
	assert groups["hygiene"]["missing_subcommands"] == []
	assert groups["studio"]["missing_subcommands"] == []
	assert report["summary"]["blocking_gap_count"] == 0


def test_docs_audit_proves_required_docs_links_and_commands():
	report = audit_docs()

	assert report["format"] == "apg.docs-audit.v1"
	assert report["ok"] is True
	assert report["summary"]["missing_required_doc_count"] == 0
	assert report["summary"]["broken_local_link_count"] == 0
	assert report["summary"]["unknown_documented_command_count"] == 0
	commands = {item["command"] for item in report["documented_commands"]}
	assert {"compile", "tooling", "doctor", "hygiene"}.issubset(commands)


def test_studio_designer_audit_proves_snapshot_and_dry_run_edit():
	report = audit_studio_designer_surface()

	assert report["format"] == "apg.studio-surface-audit.v1"
	assert report["ok"] is True
	checks = {check["name"]: check for check in report["checks"]}
	assert checks["snapshot"]["ok"] is True
	assert checks["snapshot"]["panel_count"] >= 8
	assert checks["plan_edit"]["ok"] is True
	assert checks["plan_edit"]["changed"] is True
	assert checks["plan_edit"]["written"] is False
