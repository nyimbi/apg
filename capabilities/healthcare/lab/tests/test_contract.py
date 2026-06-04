"""Tests for healthcare_lab capability contract."""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lab.capability_contract import (
	CAPABILITY_ID, PROVIDES, REQUIRES, RULES, THEME, UI_ROUTES,
	evaluate_capability_rules, get_capability_contract,
)


def test_contract_keys():
	c = get_capability_contract("lab_001")
	for key in ("capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "streaming", "provides", "requires"):
		assert key in c


def test_capability_id():
	assert get_capability_contract()["capability"] == "healthcare_lab"


def test_tenant_propagated():
	c = get_capability_contract("lab_x")
	assert c["configuration"]["tenant_id"] == "lab_x"


def test_rules_min_20():
	assert len(RULES) >= 20


def test_ui_routes_min_8():
	assert len(UI_ROUTES) >= 8
	for r in UI_ROUTES:
		for k in ("name", "path", "component", "permission", "nav_group"):
			assert k in r


def test_theme_tokens():
	for t in ("color.primary", "color.accent", "color.success", "color.warning", "color.danger", "surface.canvas", "surface.panel", "text.primary", "text.secondary", "border.radius", "density"):
		assert t in THEME["tokens"]


def test_theme_components():
	for _, comp in THEME["components"].items():
		assert "icon" in comp and "status_indicator" in comp


def test_provides_min_5():
	assert len(PROVIDES) >= 5


def test_requires_auth_audl():
	assert "auth" in REQUIRES and "audl" in REQUIRES


def test_streaming():
	c = get_capability_contract()
	s = c["streaming"]
	assert s["processor"] == "bytewax" and s["key"] == "tenant_id"
	assert len(s["events"]) >= 5


def test_deny_tenant_context():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny" and r["rule"] == "tenant_context_required"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"cross_tenant_access": True})
	assert r["decision"] == "deny"


def test_deny_critical_value_no_notification():
	r = evaluate_capability_rules({"operation": "verify_result", "critical_value": True, "notification_sent": False})
	assert r["decision"] == "deny" and "critical" in r["rule"]


def test_deny_specimen_no_rejection_reason():
	r = evaluate_capability_rules({"operation": "reject_specimen", "rejection_reason_present": False})
	assert r["decision"] == "deny"


def test_deny_qc_hold_blocks_result():
	r = evaluate_capability_rules({"operation": "verify_result", "instrument_qc_status": "qc_hold"})
	assert r["decision"] == "deny"


def test_deny_cancelled_order_collection():
	r = evaluate_capability_rules({"operation": "collect_specimen", "order_status": "cancelled"})
	assert r["decision"] == "deny"


def test_allow_default():
	r = evaluate_capability_rules({"operation": "view_dashboard"})
	assert r["decision"] == "allow" and r["rule"] is None


def test_deepcopy_isolation():
	c1 = get_capability_contract("a")
	c2 = get_capability_contract("b")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "b"
