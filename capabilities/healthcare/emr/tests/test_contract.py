"""Tests for healthcare_emr capability contract."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from emr.capability_contract import (
	CAPABILITY_ID, PROVIDES, REQUIRES, RULES, THEME, UI_ROUTES,
	evaluate_capability_rules, get_capability_contract,
)


def test_contract_keys():
	c = get_capability_contract("hosp_001")
	for key in ("capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "streaming", "provides", "requires"):
		assert key in c, f"missing key: {key}"


def test_capability_id():
	assert get_capability_contract()["capability"] == "healthcare_emr"


def test_tenant_id_in_config():
	c = get_capability_contract("tenant_x")
	assert c["configuration"]["tenant_id"] == "tenant_x"


def test_rule_engine_has_20_plus_rules():
	c = get_capability_contract()
	assert len(c["rule_engine"]["rules"]) >= 20


def test_ui_routes_min_8():
	assert len(UI_ROUTES) >= 8
	for r in UI_ROUTES:
		assert "name" in r and "path" in r and "component" in r and "permission" in r


def test_theme_required_tokens():
	tokens = THEME["tokens"]
	for t in ("color.primary", "color.accent", "color.success", "color.warning", "color.danger", "surface.canvas", "surface.panel", "text.primary", "text.secondary", "border.radius", "density"):
		assert t in tokens, f"missing token: {t}"


def test_theme_components():
	for _name, comp in THEME["components"].items():
		assert "icon" in comp and "status_indicator" in comp


def test_provides_min_5():
	assert len(PROVIDES) >= 5


def test_requires_includes_auth_audl():
	assert "auth" in REQUIRES
	assert "audl" in REQUIRES


def test_streaming_structure():
	c = get_capability_contract()
	s = c["streaming"]
	assert s["processor"] == "bytewax"
	assert s["key"] == "tenant_id"
	assert len(s["events"]) >= 5


def test_deny_tenant_context():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"
	assert r["rule"] == "tenant_context_required"


def test_deny_write_no_policy():
	r = evaluate_capability_rules({"operation_type": "write", "policy_attached": False})
	assert r["decision"] == "deny"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"cross_tenant_access": True})
	assert r["decision"] == "deny"
	assert r["rule"] == "cross_tenant_record_access_denied"


def test_deny_problem_missing_icd10():
	r = evaluate_capability_rules({"operation": "add_problem", "icd10_code_present": False})
	assert r["decision"] == "deny"
	assert "icd10" in r["rule"]


def test_deny_allergy_check_not_performed():
	r = evaluate_capability_rules({"operation": "prescribe_medication", "allergy_check_performed": False})
	assert r["decision"] == "deny"


def test_deny_deceased_record_update():
	r = evaluate_capability_rules({"operation": "update_chart", "patient_deceased": True})
	assert r["decision"] == "deny"


def test_allow_default():
	r = evaluate_capability_rules({"operation": "view_dashboard", "tenant_context_present": True})
	assert r["decision"] == "allow"
	assert r["rule"] is None


def test_deepcopy_isolation():
	c1 = get_capability_contract("a")
	c2 = get_capability_contract("b")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "b"
