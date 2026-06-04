"""Tests for realestate_val capability contract."""

from __future__ import annotations

from capabilities.realestate.val.capability_contract import (
	get_capability_contract, evaluate_capability_rules,
	CAPABILITY_ID, PROVIDES, REQUIRES, UI_ROUTES, RULES, THEME, STREAMING,
)


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "realestate_val"
	assert c["display_name"] == "Property Valuation"


def test_tenant_isolation():
	assert get_capability_contract("a")["configuration"]["tenant_id"] == "a"


def test_rules_count():
	assert len(RULES) >= 20


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	for rt in UI_ROUTES:
		assert rt["permission"].startswith("realestate_val:")


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


def test_deny_no_qualified_valuer():
	r = evaluate_capability_rules({"operation": "instruct_valuation", "qualified_valuer_assigned": False})
	assert r["decision"] == "deny"


def test_deny_red_book_non_independent():
	r = evaluate_capability_rules({"operation": "publish_valuation", "report_type": "full_red_book", "valuer_independent": False})
	assert r["decision"] == "deny"


def test_deny_sign_off_unqualified():
	r = evaluate_capability_rules({"operation": "sign_off_valuation", "valuer_grade_approved": False})
	assert r["decision"] == "deny"


def test_deny_dcf_rate_out_of_range():
	r = evaluate_capability_rules({"operation": "run_dcf", "discount_rate_in_range": False})
	assert r["decision"] == "deny"


def test_deny_dcf_missing_params():
	r = evaluate_capability_rules({"operation": "run_dcf", "discount_rate_in_range": True, "all_dcf_parameters_present": False})
	assert r["decision"] == "deny"


def test_deny_mass_appraisal_uncalibrated():
	r = evaluate_capability_rules({"operation": "run_mass_appraisal", "model_calibrated": False})
	assert r["decision"] == "deny"


def test_deny_challenge_no_evidence():
	r = evaluate_capability_rules({"operation": "raise_challenge", "counter_evidence_present": False})
	assert r["decision"] == "deny"


def test_deny_challenge_unchallengeable_status():
	r = evaluate_capability_rules({"operation": "raise_challenge", "counter_evidence_present": True, "valuation_status_challengeable": False})
	assert r["decision"] == "deny"


def test_deny_published_valuation_modification():
	r = evaluate_capability_rules({"operation_type": "write", "valuation_status": "published"})
	assert r["decision"] == "deny"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"operation_type": "write", "cross_tenant": True})
	assert r["decision"] == "deny"
