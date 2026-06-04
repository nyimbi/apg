"""Tests for realestate_prm capability contract."""

from __future__ import annotations

from capabilities.realestate.prm.capability_contract import (
	get_capability_contract, evaluate_capability_rules,
	CAPABILITY_ID, PROVIDES, REQUIRES, UI_ROUTES, RULES, THEME, STREAMING,
)


def test_contract_shape():
	contract = get_capability_contract("t1")
	assert contract["capability"] == "realestate_prm"
	assert contract["display_name"] == "Property Management"
	required = {"capability", "display_name", "version", "configuration",
				"configuration_schema", "rule_engine", "ui", "theme", "streaming", "provides", "requires"}
	assert required.issubset(contract.keys())


def test_tenant_isolation():
	assert get_capability_contract("a")["configuration"]["tenant_id"] == "a"
	assert get_capability_contract("b")["configuration"]["tenant_id"] == "b"


def test_rule_engine():
	re = get_capability_contract()["rule_engine"]
	assert re["type"] == "deterministic"
	assert re["default_decision"] == "allow"
	assert len(re["rules"]) >= 20


def test_rules_structure():
	for r in RULES:
		assert all(k in r for k in ("name", "condition", "effect"))
		assert "decision" in r["effect"]


def test_ui_routes_count():
	assert len(UI_ROUTES) >= 8


def test_ui_routes_permission_prefix():
	for route in UI_ROUTES:
		assert route["permission"].startswith("realestate_prm:")


def test_theme_required_tokens():
	tokens = THEME["tokens"]
	for t in ("color.primary", "color.accent", "surface.panel", "border.radius", "density"):
		assert t in tokens


def test_theme_components():
	assert len(THEME["components"]) >= 5
	for v in THEME["components"].values():
		assert "icon" in v and "status_indicator" in v


def test_streaming():
	assert STREAMING["processor"] == "bytewax"
	assert len(STREAMING["events"]) >= 5


def test_provides_and_requires():
	assert len(PROVIDES) >= 5
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_schema_required_fields():
	schema = get_capability_contract()["configuration_schema"]
	for f in ("tenant_id", "ui", "theme"):
		assert f in schema["required"]


def test_allow_default():
	assert evaluate_capability_rules({"tenant_context_present": True})["decision"] == "allow"


def test_deny_no_tenant():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"
	assert r["rule"] == "tenant_context_required"


def test_deny_sold_property_modification():
	r = evaluate_capability_rules({"operation_type": "write", "property_status": "sold"})
	assert r["decision"] == "deny"


def test_deny_delete_without_board_approval():
	r = evaluate_capability_rules({"operation": "delete_property", "board_approved": False})
	assert r["decision"] == "deny"


def test_deny_distribution_no_dual_control():
	r = evaluate_capability_rules({"operation": "process_distribution", "dual_control_satisfied": False})
	assert r["decision"] == "deny"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"operation_type": "write", "cross_tenant": True})
	assert r["decision"] == "deny"


def test_deny_unsupported_property_type():
	r = evaluate_capability_rules({"operation": "register_property", "property_type_supported": False})
	assert r["decision"] == "deny"


def test_deny_missing_owner():
	r = evaluate_capability_rules({"operation": "register_property", "owner_present": False, "property_type_supported": True})
	assert r["decision"] == "deny"


def test_deny_data_room_not_logged():
	r = evaluate_capability_rules({"operation": "access_data_room", "access_logged": False})
	assert r["decision"] == "deny"


def test_deny_kpi_unverified_data():
	r = evaluate_capability_rules({"operation": "calculate_kpi", "data_verified": False})
	assert r["decision"] == "deny"
