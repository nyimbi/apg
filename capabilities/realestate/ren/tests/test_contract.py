"""Tests for realestate_ren capability contract."""

from __future__ import annotations

from capabilities.realestate.ren.capability_contract import (
	get_capability_contract, evaluate_capability_rules,
	CAPABILITY_ID, PROVIDES, REQUIRES, UI_ROUTES, RULES, THEME, STREAMING,
)


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "realestate_ren"
	assert c["display_name"] == "Rental Operations"


def test_tenant_isolation():
	assert get_capability_contract("abc")["configuration"]["tenant_id"] == "abc"


def test_rules_count():
	assert len(RULES) >= 20


def test_ui_routes_count_and_prefix():
	assert len(UI_ROUTES) >= 8
	for rt in UI_ROUTES:
		assert rt["permission"].startswith("realestate_ren:")


def test_theme_tokens():
	for t in ("color.primary", "surface.panel", "border.radius", "density"):
		assert t in THEME["tokens"]


def test_streaming():
	assert STREAMING["processor"] == "bytewax"
	assert len(STREAMING["events"]) >= 5


def test_provides_requires():
	assert len(PROVIDES) >= 5
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_allow_default():
	assert evaluate_capability_rules({})["decision"] == "allow"


def test_deny_no_tenant():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"


def test_deny_no_unit():
	r = evaluate_capability_rules({"operation": "create_tenancy", "unit_present": False})
	assert r["decision"] == "deny"


def test_deny_activation_no_deposit():
	r = evaluate_capability_rules({"operation": "activate_tenancy", "deposit_registered": False})
	assert r["decision"] == "deny"


def test_deny_activation_no_referencing():
	r = evaluate_capability_rules({"operation": "activate_tenancy", "deposit_registered": True, "referencing_complete": False})
	assert r["decision"] == "deny"


def test_deny_right_to_rent_missing():
	r = evaluate_capability_rules({
		"operation": "activate_tenancy",
		"tenancy_type": "assured_shorthold",
		"right_to_rent_checked": False,
	})
	assert r["decision"] == "deny"


def test_deny_deposit_deduction_no_evidence():
	r = evaluate_capability_rules({"operation": "deduct_from_deposit", "evidence_present": False})
	assert r["decision"] == "deny"


def test_deny_deposit_deduction_exceeds_held():
	r = evaluate_capability_rules({"operation": "deduct_from_deposit", "evidence_present": True, "deduction_exceeds_held": True})
	assert r["decision"] == "deny"


def test_deny_legal_action_below_threshold():
	r = evaluate_capability_rules({"operation": "commence_legal_action", "arrears_above_threshold": False})
	assert r["decision"] == "deny"


def test_deny_vacated_tenancy_modification():
	r = evaluate_capability_rules({"operation_type": "write", "tenancy_status": "vacated"})
	assert r["decision"] == "deny"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"operation_type": "write", "cross_tenant": True})
	assert r["decision"] == "deny"
