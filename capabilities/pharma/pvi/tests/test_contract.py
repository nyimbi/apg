"""Tests for pharma_pvi capability contract."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from capabilities.pharma.pvi.capability_contract import (
	CAPABILITY_ID, RULES, SUPPORTED_AE_SOURCES, SUPPORTED_CASE_STATUSES,
	SUPPORTED_CASE_TYPES, SUPPORTED_PSUR_TYPES, SUPPORTED_REGULATORY_DATABASES,
	SUPPORTED_SIGNAL_TYPES, UI_ROUTES, evaluate_capability_rules, get_capability_contract,
)


def test_contract_shape():
	contract = get_capability_contract("pvi_tenant")
	assert contract["capability"] == CAPABILITY_ID
	assert contract["display_name"] == "Pharmacovigilance"
	assert contract["configuration"]["tenant_id"] == "pvi_tenant"


def test_theme_tokens():
	contract = get_capability_contract()
	tokens = contract["theme"]["tokens"]
	for key in ["color.primary", "color.accent", "border.radius", "density"]:
		assert key in tokens


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	names = [r["name"] for r in UI_ROUTES]
	assert "dashboard" in names
	assert "cases" in names
	assert "signals" in names


def test_supported_constants():
	assert len(SUPPORTED_AE_SOURCES) >= 8
	assert len(SUPPORTED_CASE_TYPES) >= 5
	assert len(SUPPORTED_CASE_STATUSES) >= 5
	assert len(SUPPORTED_SIGNAL_TYPES) >= 3
	assert len(SUPPORTED_PSUR_TYPES) >= 3
	assert len(SUPPORTED_REGULATORY_DATABASES) >= 5


def test_rules_count():
	assert len(RULES) >= 20


def test_evaluate_allow():
	result = evaluate_capability_rules({"tenant_context_present": True})
	assert result["decision"] == "allow"


def test_evaluate_deny_no_tenant():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "tenant_context_required" for a in result["actions"])


def test_evaluate_deny_no_meddra():
	result = evaluate_capability_rules({"operation": "process_case", "meddra_coded": False})
	assert result["decision"] == "deny"
	assert any("meddra" in a["rule"] for a in result["actions"])


def test_evaluate_deny_no_narrative():
	result = evaluate_capability_rules({"operation": "process_case", "narrative_present": False})
	assert result["decision"] == "deny"


def test_evaluate_deny_7day_reporting():
	result = evaluate_capability_rules({
		"operation": "submit_icsr", "case_type": "susar", "within_7d": False,
	})
	assert result["decision"] == "deny"
	assert any("7day" in a["rule"] for a in result["actions"])


def test_evaluate_deny_15day_reporting():
	result = evaluate_capability_rules({
		"operation": "submit_icsr", "case_serious": True, "within_15d": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_no_e2b():
	result = evaluate_capability_rules({
		"operation": "submit_icsr", "e2b_r3_formatted": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_psur_no_ibrd():
	result = evaluate_capability_rules({"operation": "create_psur", "ibrd_attached": False})
	assert result["decision"] == "deny"


def test_streaming_events():
	contract = get_capability_contract()
	events = contract["streaming"]["events"]
	assert "ae_received" in events
	assert "signal_detected" in events
	assert "psur_submitted" in events


def test_requires_nlpc():
	contract = get_capability_contract()
	assert "nlpc" in contract["requires"]
