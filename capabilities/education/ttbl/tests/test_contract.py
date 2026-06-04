"""Tests for education_ttbl capability contract shape and rule evaluation."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from capability_contract import (
	CAPABILITY_ID, PROVIDES, REQUIRES, RULES, STREAMING, THEME, UI_ROUTES,
	evaluate_capability_rules, get_capability_contract,
)


def test_contract_returns_required_top_level_keys():
	contract = get_capability_contract("acme")
	required = {"capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "provides", "requires", "streaming"}
	assert required.issubset(contract.keys())


def test_capability_id():
	assert CAPABILITY_ID == "education_ttbl"


def test_configuration_has_tenant_id():
	contract = get_capability_contract("school_c")
	assert contract["configuration"]["tenant_id"] == "school_c"


def test_configuration_schema_required_fields():
	contract = get_capability_contract()
	schema = contract["configuration_schema"]
	assert "tenant_id" in schema["required"]
	assert "ui" in schema["required"]
	assert "theme" in schema["required"]


def test_rule_engine_shape():
	contract = get_capability_contract()
	re = contract["rule_engine"]
	assert re["type"] == "deterministic"
	assert re["default_decision"] == "allow"
	assert len(re["rules"]) >= 20


def test_ui_shell_and_routes():
	contract = get_capability_contract()
	ui = contract["ui"]
	assert ui["shell"] == "apg_python"
	assert ui["requires_theme"] is True
	assert len(ui["routes"]) >= 8
	for route in ui["routes"]:
		for key in ("name", "path", "component", "permission", "nav_group"):
			assert key in route


def test_theme_required_tokens():
	required_tokens = {
		"color.primary", "color.accent", "color.success", "color.warning", "color.danger",
		"surface.canvas", "surface.panel", "text.primary", "text.secondary",
		"border.radius", "density",
	}
	assert required_tokens.issubset(THEME["tokens"].keys())


def test_theme_has_components():
	assert len(THEME["components"]) >= 5
	for _, comp in THEME["components"].items():
		assert "icon" in comp
		assert "status_indicator" in comp


def test_streaming_shape():
	assert STREAMING["processor"] == "bytewax"
	assert len(STREAMING["events"]) >= 5
	assert len(STREAMING["guardrails"]) >= 3


def test_provides_and_requires():
	assert len(PROVIDES) >= 5
	assert len(REQUIRES) >= 4
	for req in ["auth", "audl", "mten", "conf"]:
		assert req in REQUIRES


def test_rules_have_required_shape():
	for rule in RULES:
		for key in ("name", "condition", "effect"):
			assert key in rule
		for key in ("decision", "reason", "required_action"):
			assert key in rule["effect"]


def test_tenant_context_rule_denies():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "tenant_context_required"


def test_write_without_policy_denied():
	result = evaluate_capability_rules({"operation_type": "write", "policy_attached": False})
	assert result["decision"] == "deny"


def test_publish_with_unresolved_conflicts_denied():
	result = evaluate_capability_rules({
		"operation": "publish_timetable",
		"unresolved_conflicts_present": True,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "timetable_publish_requires_zero_conflicts"


def test_publish_without_approval_denied():
	result = evaluate_capability_rules({
		"operation": "publish_timetable",
		"unresolved_conflicts_present": False,
		"approval_reference_present": False,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "timetable_publish_requires_approval"


def test_constraint_removal_requires_approval():
	result = evaluate_capability_rules({
		"operation": "remove_constraint",
		"approval_reference_present": False,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "constraint_removal_requires_approval"


def test_substitution_requires_consent():
	result = evaluate_capability_rules({
		"operation": "assign_substitution",
		"teacher_consent_recorded": False,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "substitution_requires_teacher_consent"


def test_cross_tenant_room_booking_denied():
	result = evaluate_capability_rules({
		"operation": "allocate_room",
		"room_tenant_matches_requestor_tenant": False,
	})
	assert result["decision"] == "deny"


def test_room_allocation_requires_capacity_check():
	result = evaluate_capability_rules({
		"operation": "allocate_room",
		"room_tenant_matches_requestor_tenant": True,
		"capacity_check_performed": False,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "room_capacity_check_required"


def test_unsupported_algorithm_denied():
	result = evaluate_capability_rules({
		"operation": "generate_timetable",
		"algorithm_supported": False,
	})
	assert result["decision"] == "deny"


def test_no_match_allows():
	result = evaluate_capability_rules({"no_match": True})
	assert result["decision"] == "allow"
	assert result["matched_rule"] is None


def test_ui_routes_list():
	names = {r["name"] for r in UI_ROUTES}
	assert "dashboard" in names
	assert "timetables" in names
	assert "constraints" in names
	assert "conflicts" in names


def test_contract_isolation():
	c1 = get_capability_contract("t1")
	c2 = get_capability_contract("t2")
	assert c1["configuration"]["tenant_id"] == "t1"
	assert c2["configuration"]["tenant_id"] == "t2"
