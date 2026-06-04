"""Tests for healthcare_pmt capability contract."""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pmt.capability_contract import CAPABILITY_ID, PROVIDES, REQUIRES, RULES, THEME, UI_ROUTES, evaluate_capability_rules, get_capability_contract


def test_contract_keys():
	c = get_capability_contract("pmt_001")
	for key in ("capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "streaming", "provides", "requires"):
		assert key in c

def test_capability_id():
	assert get_capability_contract()["capability"] == "healthcare_pmt"

def test_tenant_propagated():
	assert get_capability_contract("hosp_p")["configuration"]["tenant_id"] == "hosp_p"

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

def test_deny_tenant_context():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"

def test_deny_discharge_no_physician_order():
	r = evaluate_capability_rules({"operation": "discharge_patient", "physician_order_present": False})
	assert r["decision"] == "deny"

def test_deny_duplicate_mrn():
	r = evaluate_capability_rules({"operation": "register_patient", "mrn_exists": True})
	assert r["decision"] == "deny"

def test_deny_inactive_patient_admit():
	r = evaluate_capability_rules({"operation": "admit_patient", "patient_status": "inactive"})
	assert r["decision"] == "deny"

def test_deny_appointment_no_slot():
	r = evaluate_capability_rules({"operation": "schedule_appointment", "slot_available": False})
	assert r["decision"] == "deny"

def test_deny_cancel_no_reason():
	r = evaluate_capability_rules({"operation": "cancel_appointment", "reason_present": False})
	assert r["decision"] == "deny"

def test_deny_merge_no_approval():
	r = evaluate_capability_rules({"operation": "merge_patients", "approval_present": False})
	assert r["decision"] == "deny"

def test_allow_default():
	r = evaluate_capability_rules({"operation": "view_dashboard"})
	assert r["decision"] == "allow" and r["rule"] is None

def test_deepcopy_isolation():
	c1 = get_capability_contract("a")
	c2 = get_capability_contract("b")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "b"
