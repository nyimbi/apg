"""Tests for pharma_dis capability contract."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from capabilities.pharma.dis.capability_contract import (
	CAPABILITY_ID, RULES, SUPPORTED_COLD_CHAIN_CLASSIFICATIONS,
	SUPPORTED_DISTRIBUTION_CHANNELS, SUPPORTED_EXCURSION_SEVERITIES,
	SUPPORTED_GDP_STATUSES, SUPPORTED_RECALL_CLASSES, SUPPORTED_RECALL_STATUSES,
	SUPPORTED_SERIALISATION_STANDARDS, SUPPORTED_SHIPMENT_STATUSES,
	SUPPORTED_TRANSPORT_MODES, SUPPORTED_WDA_STATUSES, UI_ROUTES,
	evaluate_capability_rules, get_capability_contract,
)


def test_contract_shape():
	contract = get_capability_contract("dis_tenant")
	assert contract["capability"] == CAPABILITY_ID
	assert contract["display_name"] == "Pharmaceutical Distribution"
	assert contract["configuration"]["tenant_id"] == "dis_tenant"


def test_theme_tokens():
	tokens = get_capability_contract()["theme"]["tokens"]
	for k in ["color.primary", "border.radius", "density"]:
		assert k in tokens


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	names = [r["name"] for r in UI_ROUTES]
	assert "shipments" in names
	assert "cold_chain" in names
	assert "recalls" in names


def test_supported_constants():
	assert len(SUPPORTED_DISTRIBUTION_CHANNELS) >= 5
	assert len(SUPPORTED_COLD_CHAIN_CLASSIFICATIONS) >= 5
	assert len(SUPPORTED_SERIALISATION_STANDARDS) >= 4
	assert len(SUPPORTED_RECALL_CLASSES) >= 3
	assert len(SUPPORTED_RECALL_STATUSES) >= 4
	assert len(SUPPORTED_GDP_STATUSES) >= 4
	assert len(SUPPORTED_WDA_STATUSES) >= 4
	assert len(SUPPORTED_TRANSPORT_MODES) >= 4
	assert len(SUPPORTED_SHIPMENT_STATUSES) >= 6
	assert len(SUPPORTED_EXCURSION_SEVERITIES) >= 3


def test_rules_count():
	assert len(RULES) >= 20


def test_evaluate_allow():
	result = evaluate_capability_rules({"tenant_context_present": True})
	assert result["decision"] == "allow"


def test_evaluate_deny_wda_wholesale():
	result = evaluate_capability_rules({
		"operation": "dispatch_shipment", "channel": "wholesale", "wda_active": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_no_cold_chain_monitoring():
	result = evaluate_capability_rules({
		"operation": "dispatch_shipment", "cold_chain_product": True,
		"temperature_monitoring_active": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_recall_no_regulatory_notification():
	result = evaluate_capability_rules({
		"operation": "initiate_recall", "recall_class_supported": True,
		"regulatory_notified": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_serialisation_verification():
	result = evaluate_capability_rules({
		"operation": "receive_shipment", "serialisation_verified": False,
	})
	assert result["decision"] == "deny"


def test_streaming():
	streaming = get_capability_contract()["streaming"]
	assert "recall_initiated" in streaming["events"]
	assert "cold_chain_excursion_detected" in streaming["events"]


def test_requires_monitoring():
	assert "moni" in get_capability_contract()["requires"]
