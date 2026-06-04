"""Tests for realestate_con capability contract."""

from __future__ import annotations

from capabilities.realestate.con.capability_contract import (
	get_capability_contract, evaluate_capability_rules,
	CAPABILITY_ID, PROVIDES, REQUIRES, UI_ROUTES, RULES, THEME, STREAMING,
)


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "realestate_con"
	assert c["display_name"] == "Property Contracts"


def test_tenant_isolation():
	assert get_capability_contract("a")["configuration"]["tenant_id"] == "a"
	assert get_capability_contract("b")["configuration"]["tenant_id"] == "b"


def test_rules_count():
	assert len(RULES) >= 20


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	for rt in UI_ROUTES:
		assert rt["permission"].startswith("realestate_con:")


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


def test_deny_no_parties():
	r = evaluate_capability_rules({"operation": "create_contract", "parties_present": False})
	assert r["decision"] == "deny"


def test_deny_no_governing_law():
	r = evaluate_capability_rules({"operation": "create_contract", "governing_law_present": False})
	assert r["decision"] == "deny"


def test_deny_execution_signatures_missing():
	r = evaluate_capability_rules({"operation": "execute_contract", "all_signatures_present": False})
	assert r["decision"] == "deny"


def test_deny_execution_no_legal_review():
	r = evaluate_capability_rules({"operation": "execute_contract", "all_signatures_present": True, "legal_review_complete": False})
	assert r["decision"] == "deny"


def test_deny_blacklisted_contractor():
	r = evaluate_capability_rules({"operation": "create_contract", "contractor_grade": "blacklisted"})
	assert r["decision"] == "deny"


def test_deny_variation_inactive_contract():
	r = evaluate_capability_rules({"operation": "raise_variation", "contract_status": "active", "contract_active": False})
	assert r["decision"] == "deny"


def test_deny_large_variation_no_board():
	r = evaluate_capability_rules({"operation": "raise_variation", "amount_above_threshold": True, "board_approved": False})
	assert r["decision"] == "deny"


def test_deny_retention_no_defect_clearance():
	r = evaluate_capability_rules({"operation": "release_retention", "defect_liability_cleared": False})
	assert r["decision"] == "deny"


def test_deny_termination_no_reason():
	r = evaluate_capability_rules({"operation": "terminate_contract", "reason_present": False})
	assert r["decision"] == "deny"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"operation_type": "write", "cross_tenant": True})
	assert r["decision"] == "deny"
