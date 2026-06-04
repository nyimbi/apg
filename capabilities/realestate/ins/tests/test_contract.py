"""Tests for realestate_ins capability contract."""

from __future__ import annotations

from capabilities.realestate.ins.capability_contract import (
	get_capability_contract, evaluate_capability_rules,
	CAPABILITY_ID, PROVIDES, REQUIRES, UI_ROUTES, RULES, THEME, STREAMING,
)


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "realestate_ins"
	assert c["display_name"] == "Property Insurance"


def test_tenant_isolation():
	assert get_capability_contract("a")["configuration"]["tenant_id"] == "a"


def test_rules_count():
	assert len(RULES) >= 20


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	for rt in UI_ROUTES:
		assert rt["permission"].startswith("realestate_ins:")


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


def test_deny_no_insurer():
	r = evaluate_capability_rules({"operation": "create_policy", "insurer_present": False})
	assert r["decision"] == "deny"


def test_deny_suspended_insurer_bind():
	r = evaluate_capability_rules({"operation": "bind_policy", "insurer_grade": "suspended"})
	assert r["decision"] == "deny"


def test_deny_claim_inactive_policy():
	r = evaluate_capability_rules({"operation": "lodge_claim", "policy_active": False})
	assert r["decision"] == "deny"


def test_deny_claim_uncovered_peril():
	r = evaluate_capability_rules({"operation": "lodge_claim", "policy_active": True, "peril_covered": False})
	assert r["decision"] == "deny"


def test_deny_large_claim_no_senior():
	r = evaluate_capability_rules({"operation": "approve_claim", "amount_above_threshold": True, "senior_approved": False})
	assert r["decision"] == "deny"


def test_deny_settlement_exceeds_sum():
	r = evaluate_capability_rules({"operation": "settle_claim", "settlement_exceeds_sum_insured": True})
	assert r["decision"] == "deny"


def test_deny_endorsed_sum_exceeds_market():
	r = evaluate_capability_rules({"operation": "issue_endorsement", "endorsed_sum_exceeds_market_value": True})
	assert r["decision"] == "deny"


def test_deny_critical_gap_no_alert():
	r = evaluate_capability_rules({"operation": "analyse_gaps", "critical_gap_detected": True, "alert_sent": False})
	assert r["decision"] == "deny"


def test_deny_certificate_inactive_policy():
	r = evaluate_capability_rules({"operation": "issue_certificate", "policy_active": False})
	assert r["decision"] == "deny"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"operation_type": "write", "cross_tenant": True})
	assert r["decision"] == "deny"
