"""Tests for pharma_reg capability contract."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from capabilities.pharma.reg.capability_contract import (
	CAPABILITY_ID, RULES, SUPPORTED_APPROVAL_STATUSES, SUPPORTED_AUTHORITY_INTERACTIONS,
	SUPPORTED_DOSSIER_FORMATS, SUPPORTED_LIFECYCLE_EVENTS, SUPPORTED_PROCEDURE_TYPES,
	SUPPORTED_PRODUCT_TYPES, SUPPORTED_REGISTRATION_TYPES, SUPPORTED_REGULATORY_REGIONS,
	UI_ROUTES, evaluate_capability_rules, get_capability_contract,
)


def test_contract_shape():
	contract = get_capability_contract("reg_tenant")
	assert contract["capability"] == CAPABILITY_ID
	assert contract["display_name"] == "Product Registration"
	assert contract["configuration"]["tenant_id"] == "reg_tenant"


def test_theme_tokens():
	tokens = get_capability_contract()["theme"]["tokens"]
	for k in ["color.primary", "border.radius", "density"]:
		assert k in tokens


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	names = [r["name"] for r in UI_ROUTES]
	assert "registrations" in names
	assert "dossiers" in names
	assert "approvals" in names


def test_supported_constants():
	assert len(SUPPORTED_REGISTRATION_TYPES) >= 5
	assert len(SUPPORTED_DOSSIER_FORMATS) >= 4
	assert len(SUPPORTED_APPROVAL_STATUSES) >= 7
	assert len(SUPPORTED_AUTHORITY_INTERACTIONS) >= 5
	assert len(SUPPORTED_LIFECYCLE_EVENTS) >= 6
	assert len(SUPPORTED_PRODUCT_TYPES) >= 6
	assert len(SUPPORTED_PROCEDURE_TYPES) >= 4
	assert len(SUPPORTED_REGULATORY_REGIONS) >= 8


def test_rules_count():
	assert len(RULES) >= 20


def test_evaluate_allow():
	result = evaluate_capability_rules({"tenant_context_present": True})
	assert result["decision"] == "allow"


def test_evaluate_deny_no_dossier():
	result = evaluate_capability_rules({"operation": "submit_registration", "dossier_attached": False})
	assert result["decision"] == "deny"


def test_evaluate_deny_no_qp():
	result = evaluate_capability_rules({"operation": "submit_registration", "qp_signed_off": False})
	assert result["decision"] == "deny"


def test_evaluate_deny_distribution_without_approval():
	result = evaluate_capability_rules({
		"operation": "distribute_product", "registration_approved": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_renewal_180d():
	result = evaluate_capability_rules({
		"operation": "check_registration", "expiring_within_180d": True, "renewal_initiated": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_no_local_rep():
	result = evaluate_capability_rules({
		"operation": "submit_registration", "local_representative_present": False,
	})
	assert result["decision"] == "deny"


def test_streaming():
	streaming = get_capability_contract()["streaming"]
	assert "registration_approved" in streaming["events"]
	assert "variation_filed" in streaming["events"]


def test_requires_scheduler():
	assert "schd" in get_capability_contract()["requires"]
