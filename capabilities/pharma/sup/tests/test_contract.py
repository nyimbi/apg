"""Tests for pharma_sup capability contract."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from capabilities.pharma.sup.capability_contract import (
	CAPABILITY_ID, RULES, SUPPORTED_CMO_TYPES, SUPPORTED_CONTRACT_TYPES,
	SUPPORTED_DEMAND_METHODS, SUPPORTED_IMPORT_LICENSE_TYPES, SUPPORTED_ORDER_TYPES,
	SUPPORTED_QUALIFICATION_STATUSES, SUPPORTED_SECURITY_RISK_LEVELS,
	SUPPORTED_SUPPLY_STATUSES, SUPPORTED_SUPPLIER_TYPES, UI_ROUTES,
	evaluate_capability_rules, get_capability_contract,
)


def test_contract_shape():
	contract = get_capability_contract("sup_tenant")
	assert contract["capability"] == CAPABILITY_ID
	assert contract["display_name"] == "Pharmaceutical Supply Chain"
	assert contract["configuration"]["tenant_id"] == "sup_tenant"


def test_theme_tokens():
	tokens = get_capability_contract()["theme"]["tokens"]
	for k in ["color.primary", "border.radius", "density"]:
		assert k in tokens


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	names = [r["name"] for r in UI_ROUTES]
	assert "suppliers" in names
	assert "cmo" in names
	assert "import_licensing" in names


def test_supported_constants():
	assert len(SUPPORTED_SUPPLIER_TYPES) >= 6
	assert len(SUPPORTED_QUALIFICATION_STATUSES) >= 5
	assert len(SUPPORTED_CMO_TYPES) >= 5
	assert len(SUPPORTED_DEMAND_METHODS) >= 5
	assert len(SUPPORTED_IMPORT_LICENSE_TYPES) >= 5
	assert len(SUPPORTED_SECURITY_RISK_LEVELS) >= 4
	assert len(SUPPORTED_SUPPLY_STATUSES) >= 5
	assert len(SUPPORTED_ORDER_TYPES) >= 5
	assert len(SUPPORTED_CONTRACT_TYPES) >= 4


def test_rules_count():
	assert len(RULES) >= 20


def test_evaluate_allow():
	result = evaluate_capability_rules({"tenant_context_present": True})
	assert result["decision"] == "allow"


def test_evaluate_deny_order_not_on_asl():
	result = evaluate_capability_rules({
		"operation": "place_order", "supplier_on_asl": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_cmo_no_technical_agreement():
	result = evaluate_capability_rules({
		"operation": "activate_cmo", "technical_agreement_signed": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_import_license_required():
	result = evaluate_capability_rules({
		"operation": "import_shipment", "import_license_active": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_shortage_no_regulatory_notification():
	result = evaluate_capability_rules({
		"operation": "update_supply_status", "status": "shortage",
		"regulatory_notified": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_high_risk_no_dual_source():
	result = evaluate_capability_rules({
		"operation": "confirm_supply_plan", "risk_level": "high", "dual_sourced": False,
	})
	assert result["decision"] == "deny"


def test_streaming():
	streaming = get_capability_contract()["streaming"]
	assert "supplier_qualified" in streaming["events"]
	assert "supply_shortage_detected" in streaming["events"]


def test_requires_monitoring():
	assert "moni" in get_capability_contract()["requires"]
