"""Tests for mining_pro capability contract."""

from __future__ import annotations

import pytest

from capabilities.mining.pro.capability_contract import (
	CAPABILITY_ID,
	PROVIDES,
	REQUIRES,
	RULES,
	THEME,
	UI_ROUTES,
	evaluate_capability_rules,
	get_capability_contract,
)


def test_contract_top_level_keys():
	contract = get_capability_contract("test_tenant")
	required = {"capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "provides", "requires", "streaming"}
	assert required.issubset(contract.keys())


def test_capability_id():
	assert CAPABILITY_ID == "mining_pro"


def test_tenant_propagated():
	contract = get_capability_contract("site_a")
	assert contract["configuration"]["tenant_id"] == "site_a"


def test_rule_engine_min_rules():
	contract = get_capability_contract()
	assert len(contract["rule_engine"]["rules"]) >= 20


def test_rules_structure():
	for rule in RULES:
		assert "name" in rule
		assert "condition" in rule
		assert "effect" in rule
		assert rule["effect"]["decision"] in ("allow", "deny")


def test_ui_routes_minimum():
	assert len(UI_ROUTES) >= 8


def test_ui_routes_prefix():
	for route in UI_ROUTES:
		assert route["path"].startswith("/mining-pro/")


def test_theme_tokens_complete():
	tokens = THEME["tokens"]
	for key in ("color.primary", "color.accent", "surface.canvas", "surface.panel", "text.primary", "border.radius"):
		assert key in tokens, f"Missing token: {key}"


def test_provides_minimum():
	assert len(PROVIDES) >= 5


def test_requires_mandatory():
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_streaming_events():
	contract = get_capability_contract()
	streaming = contract["streaming"]
	assert "blast_fired" in streaming["events"]
	assert "shift_report_submitted" in streaming["events"]
	assert len(streaming["guardrails"]) >= 3


def test_evaluate_tenant_context_missing():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert "attach_tenant_context" in result["required_actions"]


def test_evaluate_blast_fire_requires_authority():
	result = evaluate_capability_rules({"operation": "fire_blast", "fire_authority_present": False})
	assert result["decision"] == "deny"
	assert "obtain_fire_authority" in result["required_actions"]


def test_evaluate_blast_design_approval():
	result = evaluate_capability_rules({"operation": "charge_blast", "blast_design_approved": False})
	assert result["decision"] == "deny"


def test_evaluate_negative_tonnes():
	result = evaluate_capability_rules({"operation": "record_production", "tonnes_negative": True})
	assert result["decision"] == "deny"
	assert "correct_tonnes_value" in result["required_actions"]


def test_evaluate_future_shift_denied():
	result = evaluate_capability_rules({"operation": "create_shift_report", "shift_in_future": True})
	assert result["decision"] == "deny"


def test_evaluate_ore_tracking_method_required():
	result = evaluate_capability_rules({"operation": "record_ore_movement", "tracking_method_present": False})
	assert result["decision"] == "deny"


def test_evaluate_grade_boundary_bypass_denied():
	result = evaluate_capability_rules({"operation": "bypass_grade_boundary", "has_override_authority": False})
	assert result["decision"] == "deny"


def test_evaluate_allow_clean_context():
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] == "allow"


def test_contract_isolation_between_tenants():
	c1 = get_capability_contract("mine_a")
	c2 = get_capability_contract("mine_b")
	assert c1["configuration"]["tenant_id"] != c2["configuration"]["tenant_id"]
