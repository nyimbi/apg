"""Contract validation tests for retail_pos capability."""

import pytest
from ..capability_contract import (
	get_capability_contract,
	evaluate_capability_rules,
	CAPABILITY_ID,
	PROVIDES,
	REQUIRES,
	UI_ROUTES,
	THEME,
	RULES,
)


def test_contract_top_level_keys():
	contract = get_capability_contract()
	for key in ("capability", "display_name", "version", "provides", "requires",
				"configuration", "configuration_schema", "rule_engine", "ui", "theme", "streaming"):
		assert key in contract, f"missing key: {key}"


def test_capability_id():
	assert get_capability_contract()["capability"] == "retail_pos"


def test_configuration_has_tenant_id():
	c = get_capability_contract("acme")
	assert c["configuration"]["tenant_id"] == "acme"


def test_configuration_schema_required_fields():
	schema = get_capability_contract()["configuration_schema"]
	assert "tenant_id" in schema["required"]
	assert "ui" in schema["required"]
	assert "theme" in schema["required"]


def test_provides_count():
	assert len(PROVIDES) >= 5


def test_requires_mandatory_caps():
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_ui_routes_count():
	assert len(UI_ROUTES) >= 8


def test_ui_routes_schema():
	for route in UI_ROUTES:
		for field in ("name", "path", "component", "permission", "nav_group"):
			assert field in route, f"route missing {field}: {route}"


def test_theme_tokens():
	tokens = THEME["tokens"]
	for key in ("color.primary", "color.accent", "color.success", "color.warning",
				"color.danger", "surface.canvas", "surface.panel", "text.primary",
				"text.secondary", "border.radius", "density"):
		assert key in tokens, f"missing token: {key}"


def test_theme_components():
	assert len(THEME["components"]) >= 5
	for name, comp in THEME["components"].items():
		assert "icon" in comp
		assert "status_indicator" in comp


def test_rule_engine_type():
	re = get_capability_contract()["rule_engine"]
	assert re["type"] == "deterministic"
	assert re["default_decision"] == "allow"


def test_rules_count():
	assert len(RULES) >= 20


def test_rules_schema():
	for rule in RULES:
		assert "name" in rule
		assert "condition" in rule
		assert "effect" in rule
		assert "decision" in rule["effect"]
		assert "reason" in rule["effect"]
		assert "required_action" in rule["effect"]


def test_streaming_keys():
	s = get_capability_contract()["streaming"]
	for key in ("processor", "stream", "key", "events", "guardrails"):
		assert key in s
	assert s["processor"] == "bytewax"
	assert s["key"] == "tenant_id"


def test_evaluate_rules_allow():
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] == "allow"


def test_evaluate_rules_deny_no_tenant():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "tenant_context_required" for a in result["actions"])


def test_evaluate_rules_deny_write_no_policy():
	result = evaluate_capability_rules({"operation_type": "write", "policy_attached": False, "tenant_context_present": True})
	assert result["decision"] == "deny"


def test_contract_isolation():
	c1 = get_capability_contract("tenant_a")
	c2 = get_capability_contract("tenant_b")
	assert c1["configuration"]["tenant_id"] == "tenant_a"
	assert c2["configuration"]["tenant_id"] == "tenant_b"
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "tenant_b"


def test_ui_shell():
	ui = get_capability_contract()["ui"]
	assert ui["shell"] == "apg_python"
	assert ui["requires_theme"] is True
