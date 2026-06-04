"""Tests for realestate_mai capability contract."""

from __future__ import annotations

from capabilities.realestate.mai.capability_contract import (
	get_capability_contract, evaluate_capability_rules,
	CAPABILITY_ID, PROVIDES, REQUIRES, UI_ROUTES, RULES, THEME, STREAMING,
)


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "realestate_mai"
	assert c["display_name"] == "Facilities Maintenance"


def test_tenant_isolation():
	assert get_capability_contract("abc")["configuration"]["tenant_id"] == "abc"


def test_rules_count():
	assert len(RULES) >= 20


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	for rt in UI_ROUTES:
		assert rt["permission"].startswith("realestate_mai:")


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


def test_deny_decommissioned_asset_work_order():
	r = evaluate_capability_rules({"operation": "raise_work_order", "asset_status": "decommissioned"})
	assert r["decision"] == "deny"


def test_deny_p1_no_contractor():
	r = evaluate_capability_rules({
		"operation": "raise_work_order",
		"priority": "p1_critical",
		"contractor_assigned": False,
	})
	assert r["decision"] == "deny"


def test_deny_contractor_no_insurance():
	r = evaluate_capability_rules({"operation": "assign_contractor", "contractor_has_valid_insurance": False})
	assert r["decision"] == "deny"


def test_deny_sla_breach_no_escalation():
	r = evaluate_capability_rules({"operation": "update_work_order", "sla_breached": True, "escalated": False})
	assert r["decision"] == "deny"


def test_deny_close_without_verification():
	r = evaluate_capability_rules({"operation": "close_work_order", "verification_complete": False})
	assert r["decision"] == "deny"


def test_deny_statutory_overdue_no_alert():
	r = evaluate_capability_rules({
		"operation": "check_inspection_status",
		"inspection_type": "statutory",
		"overdue": True,
		"alert_sent": False,
	})
	assert r["decision"] == "deny"


def test_deny_cafm_not_configured():
	r = evaluate_capability_rules({"operation": "sync_cafm", "cafm_configured": False})
	assert r["decision"] == "deny"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"operation_type": "write", "cross_tenant": True})
	assert r["decision"] == "deny"
