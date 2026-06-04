"""Tests for healthcare_pha capability contract."""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pha.capability_contract import (
	CAPABILITY_ID, PROVIDES, REQUIRES, RULES, THEME, UI_ROUTES,
	evaluate_capability_rules, get_capability_contract,
)


def test_contract_keys():
	c = get_capability_contract("pha_001")
	for key in ("capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "streaming", "provides", "requires"):
		assert key in c


def test_capability_id():
	assert get_capability_contract()["capability"] == "healthcare_pha"


def test_tenant_propagated():
	c = get_capability_contract("rx_tenant")
	assert c["configuration"]["tenant_id"] == "rx_tenant"


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


def test_deny_contraindicated_dispense():
	r = evaluate_capability_rules({"operation": "dispense", "interaction_severity": "contraindicated"})
	assert r["decision"] == "deny" and "contraindicated" in r["rule"]


def test_deny_dispense_without_pharmacist():
	r = evaluate_capability_rules({"operation": "dispense", "pharmacist_verified": False})
	assert r["decision"] == "deny"


def test_deny_recalled_drug():
	r = evaluate_capability_rules({"operation": "dispense", "drug_inventory_status": "recalled"})
	assert r["decision"] == "deny"


def test_deny_expired_drug():
	r = evaluate_capability_rules({"operation": "dispense", "drug_inventory_status": "expired"})
	assert r["decision"] == "deny"


def test_deny_waste_without_witness():
	r = evaluate_capability_rules({"operation": "waste_controlled_substance", "dual_witness_present": False})
	assert r["decision"] == "deny"


def test_deny_non_formulary_without_override():
	r = evaluate_capability_rules({"operation": "dispense", "formulary_status": "non_formulary", "formulary_override_present": False})
	assert r["decision"] == "deny"


def test_deny_prior_auth_required():
	r = evaluate_capability_rules({"operation": "dispense", "formulary_status": "prior_auth_required", "prior_auth_approved": False})
	assert r["decision"] == "deny"


def test_allow_default():
	r = evaluate_capability_rules({"operation": "view_dashboard"})
	assert r["decision"] == "allow" and r["rule"] is None


def test_deepcopy_isolation():
	c1 = get_capability_contract("a")
	c2 = get_capability_contract("b")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "b"
