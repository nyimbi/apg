"""Tests for realestate_lea capability contract."""

from __future__ import annotations

from capabilities.realestate.lea.capability_contract import (
	get_capability_contract, evaluate_capability_rules,
	CAPABILITY_ID, PROVIDES, REQUIRES, UI_ROUTES, RULES, THEME, STREAMING,
)


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "realestate_lea"
	assert c["display_name"] == "Lease Management"
	assert {"capability", "display_name", "version", "configuration",
			"configuration_schema", "rule_engine", "ui", "theme", "streaming",
			"provides", "requires"}.issubset(c.keys())


def test_tenant_isolation():
	assert get_capability_contract("x")["configuration"]["tenant_id"] == "x"


def test_rules_count_and_structure():
	assert len(RULES) >= 20
	for r in RULES:
		assert all(k in r for k in ("name", "condition", "effect"))


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	for rt in UI_ROUTES:
		assert rt["permission"].startswith("realestate_lea:")


def test_theme_tokens():
	for t in ("color.primary", "surface.panel", "border.radius", "density"):
		assert t in THEME["tokens"]


def test_theme_components():
	assert len(THEME["components"]) >= 5


def test_streaming():
	assert STREAMING["processor"] == "bytewax"
	assert len(STREAMING["events"]) >= 8


def test_provides_requires():
	assert len(PROVIDES) >= 5
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_allow_default():
	assert evaluate_capability_rules({})["decision"] == "allow"


def test_deny_no_tenant():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"
	assert r["rule"] == "tenant_context_required"


def test_deny_no_property_for_lease():
	r = evaluate_capability_rules({"operation": "create_lease", "property_present": False})
	assert r["decision"] == "deny"


def test_deny_no_tenant_for_lease():
	r = evaluate_capability_rules({"operation": "create_lease", "tenant_present": False})
	assert r["decision"] == "deny"


def test_deny_activation_no_commencement():
	r = evaluate_capability_rules({"operation": "activate_lease", "commencement_date_present": False})
	assert r["decision"] == "deny"


def test_deny_ifrs16_no_discount_rate():
	r = evaluate_capability_rules({"operation": "generate_ifrs16_schedule", "discount_rate_present": False})
	assert r["decision"] == "deny"


def test_deny_option_no_notice():
	r = evaluate_capability_rules({"operation": "exercise_option", "notice_served": False})
	assert r["decision"] == "deny"


def test_deny_option_outside_window():
	r = evaluate_capability_rules({"operation": "exercise_option", "notice_served": True, "within_exercise_window": False})
	assert r["decision"] == "deny"


def test_deny_assignment_no_landlord_consent():
	r = evaluate_capability_rules({"operation": "assign_lease", "landlord_consent_obtained": False})
	assert r["decision"] == "deny"


def test_deny_forfeiture_no_legal():
	r = evaluate_capability_rules({"operation": "forfeit_lease", "legal_process_complete": False})
	assert r["decision"] == "deny"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"operation_type": "write", "cross_tenant": True})
	assert r["decision"] == "deny"


def test_deny_ifrs16_reclassification_no_auditor():
	r = evaluate_capability_rules({"operation": "reclassify_ifrs16", "auditor_approved": False})
	assert r["decision"] == "deny"
