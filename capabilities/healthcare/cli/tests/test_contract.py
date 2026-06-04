"""Tests for healthcare_cli capability contract."""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cli.capability_contract import CAPABILITY_ID, PROVIDES, REQUIRES, RULES, THEME, UI_ROUTES, evaluate_capability_rules, get_capability_contract


def test_contract_keys():
	c = get_capability_contract("cli_001")
	for key in ("capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "streaming", "provides", "requires"):
		assert key in c

def test_capability_id():
	assert get_capability_contract()["capability"] == "healthcare_cli"

def test_tenant_propagated():
	assert get_capability_contract("hosp_x")["configuration"]["tenant_id"] == "hosp_x"

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

def test_deny_activate_care_plan_no_team():
	r = evaluate_capability_rules({"operation": "activate_care_plan", "team_member_assigned": False})
	assert r["decision"] == "deny"

def test_deny_protocol_no_criteria():
	r = evaluate_capability_rules({"operation": "activate_protocol", "activation_criteria_met": False})
	assert r["decision"] == "deny"

def test_deny_handoff_no_structured_format():
	r = evaluate_capability_rules({"operation": "record_handoff", "structured_format_used": False})
	assert r["decision"] == "deny"

def test_deny_revoked_care_plan_edit():
	r = evaluate_capability_rules({"operation": "update_care_plan", "care_plan_status": "revoked"})
	assert r["decision"] == "deny"

def test_allow_default():
	r = evaluate_capability_rules({"operation": "view_dashboard"})
	assert r["decision"] == "allow" and r["rule"] is None

def test_deepcopy_isolation():
	c1 = get_capability_contract("a")
	c2 = get_capability_contract("b")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "b"
