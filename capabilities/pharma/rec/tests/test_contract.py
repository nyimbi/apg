"""Tests for pharma_rec capability contract."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from capabilities.pharma.rec.capability_contract import (
	CAPABILITY_ID, RULES, SUPPORTED_AUDIT_TYPES, SUPPORTED_COMMITMENT_STATUSES,
	SUPPORTED_INSPECTION_OUTCOMES, SUPPORTED_INTEL_TYPES, SUPPORTED_LABEL_CHANGE_TYPES,
	SUPPORTED_PMS_TYPES, SUPPORTED_REGULATORY_FRAMEWORKS, SUPPORTED_REGULATORY_REGIONS,
	UI_ROUTES, evaluate_capability_rules, get_capability_contract,
)


def test_contract_shape():
	contract = get_capability_contract("rec_tenant")
	assert contract["capability"] == CAPABILITY_ID
	assert contract["display_name"] == "Regulatory Compliance"
	assert contract["configuration"]["tenant_id"] == "rec_tenant"


def test_theme_tokens():
	tokens = get_capability_contract()["theme"]["tokens"]
	for k in ["color.primary", "border.radius", "density"]:
		assert k in tokens


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	names = [r["name"] for r in UI_ROUTES]
	assert "dashboard" in names
	assert "inspections" in names
	assert "commitments" in names


def test_supported_constants():
	assert len(SUPPORTED_REGULATORY_FRAMEWORKS) >= 8
	assert len(SUPPORTED_AUDIT_TYPES) >= 4
	assert len(SUPPORTED_INSPECTION_OUTCOMES) >= 4
	assert len(SUPPORTED_LABEL_CHANGE_TYPES) >= 3
	assert len(SUPPORTED_PMS_TYPES) >= 3
	assert len(SUPPORTED_INTEL_TYPES) >= 4
	assert len(SUPPORTED_COMMITMENT_STATUSES) >= 4
	assert len(SUPPORTED_REGULATORY_REGIONS) >= 6


def test_rules_count():
	assert len(RULES) >= 20


def test_evaluate_allow():
	result = evaluate_capability_rules({"tenant_context_present": True})
	assert result["decision"] == "allow"


def test_evaluate_deny_no_tenant():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"


def test_evaluate_deny_warning_letter_deadline():
	result = evaluate_capability_rules({
		"operation": "respond_to_inspection", "outcome": "warning_letter", "within_30d": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_label_no_qp():
	result = evaluate_capability_rules({
		"operation": "approve_label", "qp_approved": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_overdue_commitment():
	result = evaluate_capability_rules({
		"operation": "check_commitment", "overdue": True, "escalated": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_no_impact_assessment():
	result = evaluate_capability_rules({
		"operation": "record_regulatory_change", "impact_assessed": False,
	})
	assert result["decision"] == "deny"


def test_streaming():
	streaming = get_capability_contract()["streaming"]
	assert "inspection_completed" in streaming["events"]
	assert "commitment_overdue" in streaming["events"]


def test_requires_workflow():
	assert "wflo" in get_capability_contract()["requires"]
