"""Tests for MCO capability contract shape, rule evaluation, and UI routes."""

from __future__ import annotations

import sys
import os

_CAP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CAP_DIR)

from capability_contract import (
	CAPABILITY_ID,
	PROVIDES,
	REQUIRES,
	RULES,
	SUPPORTED_COMPLIANCE_DOMAINS,
	SUPPORTED_CURRENCIES,
	SUPPORTED_ENTITY_TYPES,
	SUPPORTED_INTERCOMPANY_TYPES,
	SUPPORTED_JURISDICTIONS,
	SUPPORTED_REGULATORY_FRAMEWORKS,
	SUPPORTED_STATUTORY_REPORT_TYPES,
	SUPPORTED_TRANSFER_PRICING_METHODS,
	UI_ROUTES,
	evaluate_capability_rules,
	get_capability_contract,
)


def test_capability_id():
	assert CAPABILITY_ID == "loc_mco"


def test_contract_top_level_keys():
	contract = get_capability_contract("test_tenant")
	required_keys = {"capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "streaming", "provides", "requires"}
	assert required_keys.issubset(set(contract.keys()))


def test_contract_tenant_id_propagated():
	contract = get_capability_contract("acme_corp")
	assert contract["configuration"]["tenant_id"] == "acme_corp"


def test_configuration_schema_required_keys():
	contract = get_capability_contract()
	schema = contract["configuration_schema"]
	assert "tenant_id" in schema["required"]
	assert "ui" in schema["required"]
	assert "theme" in schema["required"]


def test_rule_engine_structure():
	contract = get_capability_contract()
	re = contract["rule_engine"]
	assert re["type"] == "deterministic"
	assert re["default_decision"] == "allow"
	assert isinstance(re["rules"], list)
	assert len(re["rules"]) >= 20


def test_rules_have_required_fields():
	for rule in RULES:
		assert "name" in rule
		assert "condition" in rule
		assert "effect" in rule
		assert "decision" in rule["effect"]
		assert "reason" in rule["effect"]
		assert "required_action" in rule["effect"]


def test_provides_minimum_count():
	assert len(PROVIDES) >= 5


def test_requires_includes_core():
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_ui_routes_minimum_count():
	assert len(UI_ROUTES) >= 8


def test_ui_routes_structure():
	for route in UI_ROUTES:
		assert "name" in route
		assert "path" in route
		assert "component" in route
		assert "permission" in route
		assert "nav_group" in route


def test_theme_tokens():
	contract = get_capability_contract()
	tokens = contract["theme"]["tokens"]
	required_tokens = {
		"color.primary", "color.accent", "color.success", "color.warning", "color.danger",
		"surface.canvas", "surface.panel", "text.primary", "text.secondary",
		"border.radius", "density",
	}
	assert required_tokens.issubset(set(tokens.keys()))


def test_theme_components_exist():
	contract = get_capability_contract()
	components = contract["theme"]["components"]
	assert len(components) >= 5
	for comp in components.values():
		assert "icon" in comp
		assert "status_indicator" in comp


def test_streaming_structure():
	contract = get_capability_contract()
	s = contract["streaming"]
	assert s["processor"] == "bytewax"
	assert "stream" in s
	assert "key" in s
	assert isinstance(s["events"], list)
	assert len(s["events"]) >= 5
	assert isinstance(s["guardrails"], list)


def test_evaluate_allow_clean_context():
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] == "allow"
	assert result["actions"] == []


def test_evaluate_deny_missing_tenant():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "tenant_context_required" for a in result["actions"])


def test_evaluate_deny_write_no_policy():
	result = evaluate_capability_rules({"operation_type": "write", "policy_attached": False, "tenant_context_present": True})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "write_requires_policy" for a in result["actions"])


def test_evaluate_deny_unsupported_jurisdiction():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "register_country",
		"jurisdiction_supported": False,
		"currency_supported": True,
		"regulatory_framework_present": True,
		"country_name_present": True,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "country_jurisdiction_supported" for a in result["actions"])


def test_evaluate_deny_arms_length_bypass():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "create_intercompany",
		"transaction_type_supported": True,
		"originator_present": True,
		"counterparty_present": True,
		"currency_supported": True,
		"arms_length_bypass": True,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "arms_length_bypass_denied" for a in result["actions"])


def test_evaluate_deny_overdue_report():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "create_statutory_report",
		"report_type_supported": True,
		"entity_present": True,
		"period_present": True,
		"existing_overdue_unfiled": True,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "overdue_report_filing_blocked" for a in result["actions"])


def test_evaluate_deny_privileged_agent_no_approval():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "agent_action",
		"privileged_scope": True,
		"human_approval_recorded": False,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "privileged_agent_action_requires_human_approval" for a in result["actions"])


def test_supported_constants_nonempty():
	assert len(SUPPORTED_JURISDICTIONS) >= 5
	assert len(SUPPORTED_CURRENCIES) >= 5
	assert len(SUPPORTED_ENTITY_TYPES) >= 3
	assert len(SUPPORTED_REGULATORY_FRAMEWORKS) >= 3
	assert len(SUPPORTED_INTERCOMPANY_TYPES) >= 5
	assert len(SUPPORTED_STATUTORY_REPORT_TYPES) >= 5
	assert len(SUPPORTED_TRANSFER_PRICING_METHODS) >= 3
	assert len(SUPPORTED_COMPLIANCE_DOMAINS) >= 5


def test_contract_is_deepcopy():
	c1 = get_capability_contract("t1")
	c2 = get_capability_contract("t2")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "t2"
