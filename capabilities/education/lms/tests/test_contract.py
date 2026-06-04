"""Tests for education_lms capability contract shape and rule evaluation."""

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
	assert CAPABILITY_ID == "education_lms"


def test_configuration_has_tenant_id():
	contract = get_capability_contract("school_a")
	assert contract["configuration"]["tenant_id"] == "school_a"


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
	assert isinstance(re["rules"], list)
	assert len(re["rules"]) >= 20


def test_ui_shell_and_routes():
	contract = get_capability_contract()
	ui = contract["ui"]
	assert ui["shell"] == "apg_python"
	assert ui["requires_theme"] is True
	assert len(ui["routes"]) >= 8
	for route in ui["routes"]:
		assert "name" in route
		assert "path" in route
		assert "component" in route
		assert "permission" in route
		assert "nav_group" in route


def test_theme_required_tokens():
	required_tokens = {
		"color.primary", "color.accent", "color.success", "color.warning", "color.danger",
		"surface.canvas", "surface.panel", "text.primary", "text.secondary",
		"border.radius", "density",
	}
	assert required_tokens.issubset(THEME["tokens"].keys())


def test_theme_has_components():
	assert len(THEME["components"]) >= 5
	for name, comp in THEME["components"].items():
		assert "icon" in comp
		assert "status_indicator" in comp


def test_streaming_shape():
	assert STREAMING["processor"] == "bytewax"
	assert "stream" in STREAMING
	assert "key" in STREAMING
	assert len(STREAMING["events"]) >= 5
	assert len(STREAMING["guardrails"]) >= 3


def test_provides_and_requires():
	assert len(PROVIDES) >= 5
	assert len(REQUIRES) >= 4
	for req in ["auth", "audl", "mten", "conf"]:
		assert req in REQUIRES


def test_rules_have_required_shape():
	for rule in RULES:
		assert "name" in rule
		assert "condition" in rule
		assert "effect" in rule
		assert "decision" in rule["effect"]
		assert "reason" in rule["effect"]
		assert "required_action" in rule["effect"]


def test_tenant_context_rule_denies():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "tenant_context_required"


def test_write_without_policy_denied():
	result = evaluate_capability_rules({"operation_type": "write", "policy_attached": False})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "lms_write_requires_policy"


def test_unsupported_course_type_denied():
	result = evaluate_capability_rules({
		"operation": "create_course",
		"course_type_supported": False,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "course_type_supported"


def test_publish_without_review_denied():
	result = evaluate_capability_rules({
		"operation": "publish_course",
		"review_approved": False,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "course_publish_requires_review"


def test_paid_enrolment_without_payment_denied():
	result = evaluate_capability_rules({
		"operation": "enrol_learner",
		"enrolment_type": "paid",
		"payment_reference_present": False,
	})
	assert result["decision"] == "deny"


def test_grade_override_without_approval_denied():
	result = evaluate_capability_rules({
		"operation": "override_grade",
		"approval_reference_present": False,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "grade_override_requires_approval"


def test_certificate_without_completion_denied():
	result = evaluate_capability_rules({
		"operation": "issue_certificate",
		"completion_criteria_met": False,
		"certificate_type_supported": True,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "certificate_requires_completion"


def test_no_match_allows():
	result = evaluate_capability_rules({"completely_irrelevant_key": True})
	assert result["decision"] == "allow"
	assert result["matched_rule"] is None


def test_analytics_export_requires_consent():
	result = evaluate_capability_rules({
		"operation": "export_learner_analytics",
		"consent_recorded": False,
	})
	assert result["decision"] == "deny"


def test_ui_routes_list():
	assert len(UI_ROUTES) >= 8
	names = {r["name"] for r in UI_ROUTES}
	assert "dashboard" in names
	assert "courses" in names
	assert "enrolments" in names
	assert "gradebook" in names


def test_contract_isolation_between_tenants():
	c1 = get_capability_contract("tenant_a")
	c2 = get_capability_contract("tenant_b")
	assert c1["configuration"]["tenant_id"] == "tenant_a"
	assert c2["configuration"]["tenant_id"] == "tenant_b"
