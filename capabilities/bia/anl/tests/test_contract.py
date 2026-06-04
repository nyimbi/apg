"""Tests for bia_anl capability contract."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capability_contract import (
	CAPABILITY_ID, PROVIDES, REQUIRES, UI_ROUTES, THEME, STREAMING, RULES,
	get_capability_contract, evaluate_capability_rules,
)


def test_capability_id():
	assert CAPABILITY_ID == "bia_anl"


def test_contract_top_level_keys():
	c = get_capability_contract("acme")
	for key in ["capability", "display_name", "version", "configuration", "configuration_schema",
	            "rule_engine", "ui", "theme", "streaming", "provides", "requires"]:
		assert key in c, f"missing key: {key}"


def test_configuration_has_tenant_id():
	c = get_capability_contract("acme")
	assert c["configuration"]["tenant_id"] == "acme"


def test_configuration_schema_required_fields():
	c = get_capability_contract()
	req = c["configuration_schema"]["required"]
	for f in ["tenant_id", "ui", "theme"]:
		assert f in req


def test_rule_engine_structure():
	c = get_capability_contract()
	re = c["rule_engine"]
	assert re["type"] == "deterministic"
	assert re["default_decision"] == "allow"
	assert isinstance(re["rules"], list)
	assert len(re["rules"]) >= 20


def test_ui_shell_and_routes():
	c = get_capability_contract()
	ui = c["ui"]
	assert ui["shell"] == "apg_python"
	assert ui["requires_theme"] is True
	assert len(ui["routes"]) >= 8


def test_theme_tokens():
	tokens = THEME["tokens"]
	for key in ["color.primary", "color.accent", "color.success", "color.warning", "color.danger",
	            "surface.canvas", "surface.panel", "text.primary", "text.secondary", "border.radius", "density"]:
		assert key in tokens, f"missing token: {key}"


def test_provides_count():
	assert len(PROVIDES) >= 5


def test_requires_contains_core():
	for cap in ["auth", "audl", "mten", "conf"]:
		assert cap in REQUIRES


def test_streaming_structure():
	assert STREAMING["processor"] == "bytewax"
	assert "stream" in STREAMING
	assert "key" in STREAMING
	assert len(STREAMING["events"]) >= 5
	assert len(STREAMING["guardrails"]) >= 3


def test_rules_have_required_fields():
	for rule in RULES:
		assert "name" in rule
		assert "condition" in rule
		assert "effect" in rule
		assert "decision" in rule["effect"]
		assert "reason" in rule["effect"]
		assert "required_action" in rule["effect"]


def test_evaluate_deny_on_missing_tenant():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert result["matched_rule"] == "tenant_context_required"


def test_evaluate_deny_cross_tenant():
	result = evaluate_capability_rules({"cross_tenant_access": True})
	assert result["decision"] == "deny"


def test_evaluate_allow_no_match():
	result = evaluate_capability_rules({"operation": "list_metrics", "tenant_context_present": True})
	assert result["decision"] == "allow"
	assert result["matched_rule"] is None


def test_ui_routes_have_required_fields():
	for route in UI_ROUTES:
		for field in ["name", "path", "component", "permission", "nav_group"]:
			assert field in route, f"route missing {field}: {route}"


def test_theme_components_not_empty():
	assert len(THEME["components"]) >= 3
	for name, comp in THEME["components"].items():
		assert "icon" in comp
		assert "status_indicator" in comp
