"""Tests for pharma_ctr capability contract."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from capabilities.pharma.ctr.capability_contract import (
	CAPABILITY_ID, RULES, SUPPORTED_AE_SEVERITIES, SUPPORTED_AE_TYPES,
	SUPPORTED_BLINDING_TYPES, SUPPORTED_PATIENT_STATUSES, SUPPORTED_PROTOCOL_STATUSES,
	SUPPORTED_RANDOMISATION_METHODS, SUPPORTED_REGULATORY_AUTHORITIES, SUPPORTED_SITE_STATUSES,
	SUPPORTED_SUBMISSION_TYPES, SUPPORTED_TRIAL_PHASES, SUPPORTED_TRIAL_TYPES,
	UI_ROUTES, evaluate_capability_rules, get_capability_contract,
)


def test_contract_shape():
	contract = get_capability_contract("ctr_tenant")
	assert contract["capability"] == CAPABILITY_ID
	assert contract["display_name"] == "Clinical Trials Management"
	assert contract["configuration"]["tenant_id"] == "ctr_tenant"


def test_theme_tokens():
	tokens = get_capability_contract()["theme"]["tokens"]
	for k in ["color.primary", "color.accent", "border.radius", "density"]:
		assert k in tokens


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	names = [r["name"] for r in UI_ROUTES]
	assert "dashboard" in names
	assert "trials" in names
	assert "adverse_events" in names


def test_supported_constants():
	assert len(SUPPORTED_TRIAL_PHASES) >= 7
	assert len(SUPPORTED_TRIAL_TYPES) >= 5
	assert len(SUPPORTED_SITE_STATUSES) >= 6
	assert len(SUPPORTED_PATIENT_STATUSES) >= 6
	assert len(SUPPORTED_AE_SEVERITIES) >= 4
	assert len(SUPPORTED_AE_TYPES) >= 4
	assert len(SUPPORTED_SUBMISSION_TYPES) >= 5
	assert len(SUPPORTED_RANDOMISATION_METHODS) >= 4
	assert len(SUPPORTED_BLINDING_TYPES) >= 3
	assert len(SUPPORTED_REGULATORY_AUTHORITIES) >= 5
	assert len(SUPPORTED_PROTOCOL_STATUSES) >= 4


def test_rules_count():
	assert len(RULES) >= 20


def test_evaluate_allow():
	result = evaluate_capability_rules({"tenant_context_present": True})
	assert result["decision"] == "allow"


def test_evaluate_deny_no_irb():
	result = evaluate_capability_rules({"operation": "activate_trial", "irb_approved": False})
	assert result["decision"] == "deny"


def test_evaluate_deny_no_informed_consent():
	result = evaluate_capability_rules({
		"operation": "enrol_patient", "informed_consent_obtained": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_sadie_timeline():
	result = evaluate_capability_rules({
		"operation": "report_ae", "ae_type": "serious_adverse_event", "within_24h": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_susar_timeline():
	result = evaluate_capability_rules({
		"operation": "report_ae",
		"ae_type": "suspected_unexpected_serious_adverse_reaction",
		"within_15d": False,
	})
	assert result["decision"] == "deny"


def test_streaming():
	streaming = get_capability_contract()["streaming"]
	assert "patient_randomised" in streaming["events"]
	assert "adverse_event_reported" in streaming["events"]
	assert "gcp_compliance_required" in streaming["guardrails"]


def test_requires_nlpc():
	assert "nlpc" in get_capability_contract()["requires"]
