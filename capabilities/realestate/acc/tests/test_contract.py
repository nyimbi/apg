"""Tests for realestate_acc capability contract shape, rules, and UI routes."""

from __future__ import annotations

from capabilities.realestate.acc.capability_contract import (
	get_capability_contract,
	evaluate_capability_rules,
	CAPABILITY_ID,
	PROVIDES,
	REQUIRES,
	UI_ROUTES,
	RULES,
	THEME,
	STREAMING,
)


def test_contract_shape():
	contract = get_capability_contract("test-tenant")
	assert contract["capability"] == "realestate_acc"
	assert contract["display_name"] == "Real Estate Accounting"
	assert contract["version"].count(".") == 2
	assert contract["configuration"]["tenant_id"] == "test-tenant"
	required_keys = {"capability", "display_name", "version", "configuration",
					 "configuration_schema", "rule_engine", "ui", "theme", "streaming", "provides", "requires"}
	assert required_keys.issubset(contract.keys())


def test_configuration_schema_required_fields():
	contract = get_capability_contract()
	schema = contract["configuration_schema"]
	assert "tenant_id" in schema["required"]
	assert "ui" in schema["required"]
	assert "theme" in schema["required"]


def test_tenant_id_isolation():
	c1 = get_capability_contract("tenant-a")
	c2 = get_capability_contract("tenant-b")
	assert c1["configuration"]["tenant_id"] == "tenant-a"
	assert c2["configuration"]["tenant_id"] == "tenant-b"
	assert c1["configuration"]["tenant_id"] != c2["configuration"]["tenant_id"]


def test_rule_engine_structure():
	contract = get_capability_contract()
	re = contract["rule_engine"]
	assert re["type"] == "deterministic"
	assert re["default_decision"] == "allow"
	assert isinstance(re["rules"], list)
	assert len(re["rules"]) >= 20


def test_rules_have_required_keys():
	for rule in RULES:
		assert "name" in rule
		assert "condition" in rule
		assert "effect" in rule
		assert "decision" in rule["effect"]
		assert "reason" in rule["effect"]


def test_ui_shell_and_theme():
	contract = get_capability_contract()
	assert contract["ui"]["shell"] == "apg_python"
	assert contract["ui"]["requires_theme"] is True
	assert isinstance(contract["ui"]["template_roots"], list)


def test_ui_routes_count_and_shape():
	assert len(UI_ROUTES) >= 8
	for route in UI_ROUTES:
		assert "name" in route
		assert "path" in route
		assert "component" in route
		assert "permission" in route
		assert "nav_group" in route


def test_ui_routes_all_have_correct_permission_prefix():
	for route in UI_ROUTES:
		assert route["permission"].startswith("realestate_acc:")


def test_theme_tokens():
	tokens = THEME["tokens"]
	required_tokens = {"color.primary", "color.accent", "color.success", "color.warning",
					   "color.danger", "surface.canvas", "surface.panel",
					   "text.primary", "text.secondary", "border.radius", "density"}
	assert required_tokens.issubset(tokens.keys())


def test_theme_components_not_empty():
	assert len(THEME["components"]) >= 5
	for key, val in THEME["components"].items():
		assert "icon" in val
		assert "status_indicator" in val


def test_streaming_structure():
	assert STREAMING["processor"] == "bytewax"
	assert isinstance(STREAMING["stream"], str)
	assert STREAMING["key"] == "tenant_id"
	assert len(STREAMING["events"]) >= 5
	assert len(STREAMING["guardrails"]) >= 3


def test_provides_count():
	assert len(PROVIDES) >= 5


def test_requires_always_includes_core():
	for cap in ["auth", "audl", "mten", "conf"]:
		assert cap in REQUIRES


def test_evaluate_allow_when_no_conditions_match():
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] == "allow"
	assert result["rule"] is None


def test_evaluate_deny_missing_tenant_context():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert result["rule"] == "tenant_context_required"


def test_evaluate_deny_journal_not_balanced():
	result = evaluate_capability_rules({
		"operation": "post_journal",
		"entries_balanced": False,
		"period_open": True,
	})
	assert result["decision"] == "deny"
	assert "balance" in result["reason"] or "journal" in result["reason"]


def test_evaluate_deny_period_closed():
	result = evaluate_capability_rules({
		"operation": "post_journal",
		"entries_balanced": True,
		"period_open": False,
	})
	assert result["decision"] == "deny"
	assert "period" in result["reason"].lower() or "closed" in result["reason"].lower()


def test_evaluate_deny_cross_tenant():
	result = evaluate_capability_rules({
		"operation": "post_journal",
		"cross_tenant": True,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_delete_posted_journal():
	result = evaluate_capability_rules({
		"operation": "delete_journal",
		"journal_status": "posted",
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_cam_without_approval():
	result = evaluate_capability_rules({
		"operation": "settle_cam",
		"cam_approved": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_period_close_no_dual_control():
	result = evaluate_capability_rules({
		"operation": "close_period",
		"dual_control_satisfied": False,
		"reconciliations_complete": True,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_ifrs16_no_discount_rate():
	result = evaluate_capability_rules({
		"operation": "create_ifrs16_schedule",
		"lease_term_present": True,
		"discount_rate_present": False,
	})
	assert result["decision"] == "deny"
