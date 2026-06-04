"""Tests for realestate_ten capability contract."""

from __future__ import annotations

from capabilities.realestate.ten.capability_contract import (
	get_capability_contract, evaluate_capability_rules,
	CAPABILITY_ID, PROVIDES, REQUIRES, UI_ROUTES, RULES, THEME, STREAMING,
)


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "realestate_ten"
	assert c["display_name"] == "Tenant Management"


def test_tenant_isolation():
	assert get_capability_contract("a")["configuration"]["tenant_id"] == "a"


def test_rules_count():
	assert len(RULES) >= 20


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	for rt in UI_ROUTES:
		assert rt["permission"].startswith("realestate_ten:")


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


def test_deny_blacklisted_activation():
	r = evaluate_capability_rules({"operation": "activate_tenant", "tenant_status": "blacklisted"})
	assert r["decision"] == "deny"


def test_deny_activation_incomplete_onboarding():
	r = evaluate_capability_rules({"operation": "activate_tenant", "tenant_status": "approved", "mandatory_onboarding_complete": False})
	assert r["decision"] == "deny"


def test_deny_service_request_sla_breach_no_escalation():
	r = evaluate_capability_rules({"operation": "update_service_request", "sla_breached": True, "escalated": False})
	assert r["decision"] == "deny"


def test_deny_invalid_satisfaction_rating():
	r = evaluate_capability_rules({"operation": "record_satisfaction", "rating_valid": False})
	assert r["decision"] == "deny"


def test_deny_low_score_no_review():
	r = evaluate_capability_rules({"operation": "record_satisfaction", "score_below_threshold": True, "review_triggered": False})
	assert r["decision"] == "deny"


def test_deny_retention_risk_no_review():
	r = evaluate_capability_rules({"operation": "flag_retention_risk", "account_review_scheduled": False})
	assert r["decision"] == "deny"


def test_deny_data_access_not_logged():
	r = evaluate_capability_rules({"operation": "access_tenant_data", "access_logged": False})
	assert r["decision"] == "deny"


def test_deny_onboarding_prereqs_not_met():
	r = evaluate_capability_rules({"operation": "complete_onboarding_step", "prerequisite_steps_complete": False})
	assert r["decision"] == "deny"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"operation_type": "write", "cross_tenant": True})
	assert r["decision"] == "deny"
