"""Tests for pharma_com capability contract."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from capabilities.pharma.com.capability_contract import (
	CAPABILITY_ID, CAPABILITY_VERSION, PROVIDES, REQUIRES, RULES, SUPPORTED_CALL_TYPES,
	SUPPORTED_INTERACTION_TYPES, SUPPORTED_PLAN_STATUSES, SUPPORTED_REP_TYPES,
	SUPPORTED_SAMPLE_TYPES, SUPPORTED_SPEND_CATEGORIES, SUPPORTED_TARGET_TIERS,
	SUPPORTED_TERRITORY_TYPES, UI_ROUTES, evaluate_capability_rules, get_capability_contract,
)


def test_contract_shape():
	contract = get_capability_contract("test_tenant")
	assert contract["capability"] == CAPABILITY_ID
	assert contract["display_name"] == "Commercial Operations"
	assert contract["version"] == CAPABILITY_VERSION
	assert "tenant_id" in contract["configuration"]
	assert contract["configuration"]["tenant_id"] == "test_tenant"
	assert "rule_engine" in contract
	assert contract["rule_engine"]["type"] == "deterministic"
	assert contract["rule_engine"]["default_decision"] == "allow"


def test_contract_provides():
	contract = get_capability_contract()
	assert len(contract["provides"]) >= 5
	assert "territory_management_workflow" in contract["provides"]
	assert "sample_management_workflow" in contract["provides"]


def test_contract_requires():
	contract = get_capability_contract()
	for cap in ["auth", "audl", "mten", "conf"]:
		assert cap in contract["requires"]


def test_configuration_schema():
	contract = get_capability_contract()
	schema = contract["configuration_schema"]
	assert schema["type"] == "object"
	assert "tenant_id" in schema["required"]
	assert "ui" in schema["required"]
	assert "theme" in schema["required"]


def test_theme_tokens():
	contract = get_capability_contract()
	tokens = contract["theme"]["tokens"]
	required = ["color.primary", "color.accent", "color.success", "color.warning",
				"color.danger", "surface.canvas", "surface.panel", "text.primary",
				"text.secondary", "border.radius", "density"]
	for key in required:
		assert key in tokens, f"Missing theme token: {key}"


def test_theme_components():
	contract = get_capability_contract()
	components = contract["theme"]["components"]
	assert len(components) >= 5
	for name, comp in components.items():
		assert "icon" in comp
		assert "status_indicator" in comp


def test_ui_routes():
	contract = get_capability_contract()
	routes = contract["ui"]["routes"]
	assert len(routes) >= 8
	for route in routes:
		assert "name" in route
		assert "path" in route
		assert "component" in route
		assert "permission" in route
		assert "nav_group" in route


def test_ui_shell():
	contract = get_capability_contract()
	assert contract["ui"]["shell"] == "apg_python"
	assert contract["ui"]["requires_theme"] is True


def test_streaming():
	contract = get_capability_contract()
	streaming = contract["streaming"]
	assert streaming["processor"] == "bytewax"
	assert "stream" in streaming
	assert "key" in streaming
	assert len(streaming["events"]) >= 5
	assert len(streaming["guardrails"]) >= 3


def test_supported_constants():
	assert len(SUPPORTED_TERRITORY_TYPES) >= 5
	assert len(SUPPORTED_REP_TYPES) >= 5
	assert len(SUPPORTED_CALL_TYPES) >= 5
	assert len(SUPPORTED_SAMPLE_TYPES) >= 3
	assert len(SUPPORTED_INTERACTION_TYPES) >= 5
	assert len(SUPPORTED_PLAN_STATUSES) >= 4
	assert len(SUPPORTED_TARGET_TIERS) >= 4
	assert len(SUPPORTED_SPEND_CATEGORIES) >= 5


def test_rules_count():
	assert len(RULES) >= 20


def test_evaluate_allow():
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] == "allow"
	assert result["actions"] == []


def test_evaluate_deny_no_tenant():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "tenant_context_required" for a in result["actions"])


def test_evaluate_deny_no_policy():
	result = evaluate_capability_rules({"operation_type": "write", "policy_attached": False})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "write_requires_policy" for a in result["actions"])


def test_evaluate_deny_pdma():
	result = evaluate_capability_rules({
		"operation": "dispense_sample",
		"pdma_compliant": False,
	})
	assert result["decision"] == "deny"
	deny_rules = [a["rule"] for a in result["actions"]]
	assert "sample_pdma_compliance_required" in deny_rules


def test_evaluate_deny_aggregate_cap():
	result = evaluate_capability_rules({
		"operation": "record_spend",
		"aggregate_cap_exceeded": True,
	})
	assert result["decision"] == "deny"
	assert any("aggregate_spend_cap" in a["rule"] for a in result["actions"])


def test_contract_immutable_across_tenants():
	c1 = get_capability_contract("tenant_a")
	c2 = get_capability_contract("tenant_b")
	assert c1["configuration"]["tenant_id"] == "tenant_a"
	assert c2["configuration"]["tenant_id"] == "tenant_b"
	assert c1["configuration"]["tenant_id"] != c2["configuration"]["tenant_id"]


def test_rule_effects_have_required_keys():
	for rule in RULES:
		assert "name" in rule
		assert "condition" in rule
		assert "effect" in rule
		effect = rule["effect"]
		assert "decision" in effect
		assert "reason" in effect
		assert "required_action" in effect
