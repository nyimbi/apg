"""Tests for healthcare_tel capability contract."""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tel.capability_contract import CAPABILITY_ID, PROVIDES, REQUIRES, RULES, THEME, UI_ROUTES, evaluate_capability_rules, get_capability_contract


def test_contract_keys():
	c = get_capability_contract("tel_001")
	for key in ("capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "streaming", "provides", "requires"):
		assert key in c

def test_capability_id():
	assert get_capability_contract()["capability"] == "healthcare_tel"

def test_tenant_propagated():
	assert get_capability_contract("clinic_x")["configuration"]["tenant_id"] == "clinic_x"

def test_rules_min_20():
	assert len(RULES) >= 20

def test_ui_routes_min_8():
	assert len(UI_ROUTES) >= 8

def test_theme_tokens():
	for t in ("color.primary", "color.accent", "color.success", "color.warning", "color.danger", "surface.canvas", "surface.panel", "text.primary", "text.secondary", "border.radius", "density"):
		assert t in THEME["tokens"]

def test_provides_min_5():
	assert len(PROVIDES) >= 5

def test_requires_auth_audl():
	assert "auth" in REQUIRES and "audl" in REQUIRES

def test_streaming():
	c = get_capability_contract()
	s = c["streaming"]
	assert s["processor"] == "bytewax" and s["key"] == "tenant_id"
	assert len(s["events"]) >= 5

def test_deny_session_no_consent():
	r = evaluate_capability_rules({"operation": "start_session", "patient_consent_obtained": False})
	assert r["decision"] == "deny"

def test_deny_session_no_e911():
	r = evaluate_capability_rules({"operation": "start_session", "e911_disclosure_provided": False})
	assert r["decision"] == "deny"

def test_deny_schedule_ii_no_in_person():
	r = evaluate_capability_rules({"operation": "transmit_prescription", "drug_schedule": "schedule_ii", "in_person_visit_completed": False})
	assert r["decision"] == "deny"

def test_deny_recording_no_consent():
	r = evaluate_capability_rules({"operation": "start_recording", "recording_consent_obtained": False})
	assert r["decision"] == "deny"

def test_deny_cancelled_consultation_start():
	r = evaluate_capability_rules({"operation": "start_session", "consultation_status": "cancelled"})
	assert r["decision"] == "deny"

def test_deny_monitoring_no_threshold():
	r = evaluate_capability_rules({"operation": "enroll_monitoring_device", "alert_threshold_configured": False})
	assert r["decision"] == "deny"

def test_allow_default():
	r = evaluate_capability_rules({"operation": "view_dashboard"})
	assert r["decision"] == "allow" and r["rule"] is None

def test_deepcopy_isolation():
	c1 = get_capability_contract("a")
	c2 = get_capability_contract("b")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "b"
