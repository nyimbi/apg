"""Tests for education_sch_mgmt capability contract shape and rule evaluation."""

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
	assert CAPABILITY_ID == "education_sch_mgmt"


def test_configuration_has_tenant_id():
	contract = get_capability_contract("school_b")
	assert contract["configuration"]["tenant_id"] == "school_b"


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
	assert "stream" in STREAMING
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


def test_expulsion_requires_approval():
	result = evaluate_capability_rules({
		"operation": "update_student_status",
		"new_status": "expelled",
		"approval_reference_present": False,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "expulsion_requires_approval"


def test_fee_waiver_requires_approval():
	result = evaluate_capability_rules({
		"operation": "waive_fee",
		"approval_reference_present": False,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "fee_waiver_requires_approval"


def test_fee_refund_requires_approval():
	result = evaluate_capability_rules({
		"operation": "refund_fee",
		"approval_reference_present": False,
	})
	assert result["decision"] == "deny"


def test_document_sharing_requires_consent():
	result = evaluate_capability_rules({
		"operation": "share_document",
		"consent_recorded": False,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "document_sharing_requires_consent"


def test_cross_tenant_access_denied():
	result = evaluate_capability_rules({
		"operation": "access_student_record",
		"record_tenant_matches_requestor_tenant": False,
	})
	assert result["decision"] == "deny"


def test_admission_offer_requires_capacity():
	result = evaluate_capability_rules({
		"operation": "offer_admission",
		"capacity_available": False,
	})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "admission_offer_requires_capacity_check"


def test_unsupported_student_status_denied():
	result = evaluate_capability_rules({
		"operation": "update_student_status",
		"student_status_supported": False,
	})
	assert result["decision"] == "deny"


def test_no_match_allows():
	result = evaluate_capability_rules({"completely_irrelevant": True})
	assert result["decision"] == "allow"
	assert result["matched_rule"] is None


def test_ui_routes_list():
	names = {r["name"] for r in UI_ROUTES}
	assert "dashboard" in names
	assert "students" in names
	assert "admissions" in names
	assert "fees" in names


def test_contract_isolation_between_tenants():
	c1 = get_capability_contract("t1")
	c2 = get_capability_contract("t2")
	assert c1["configuration"]["tenant_id"] == "t1"
	assert c2["configuration"]["tenant_id"] == "t2"
