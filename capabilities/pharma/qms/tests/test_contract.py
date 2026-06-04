"""Tests for pharma_qms capability contract."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from capabilities.pharma.qms.capability_contract import (
	CAPABILITY_ID, RULES, SUPPORTED_AUDIT_TYPES, SUPPORTED_CAPA_TYPES,
	SUPPORTED_CHANGE_TYPES, SUPPORTED_DEVIATION_TYPES, SUPPORTED_DOCUMENT_TYPES,
	SUPPORTED_VALIDATION_TYPES, UI_ROUTES, evaluate_capability_rules, get_capability_contract,
)


def test_contract_shape():
	contract = get_capability_contract("qms_test")
	assert contract["capability"] == CAPABILITY_ID
	assert contract["display_name"] == "Quality Management System"
	assert contract["configuration"]["tenant_id"] == "qms_test"
	assert contract["rule_engine"]["default_decision"] == "allow"


def test_theme_tokens_complete():
	contract = get_capability_contract()
	tokens = contract["theme"]["tokens"]
	for key in ["color.primary", "color.accent", "color.success", "color.warning",
				"color.danger", "surface.canvas", "surface.panel", "text.primary",
				"text.secondary", "border.radius", "density"]:
		assert key in tokens


def test_ui_routes_count():
	assert len(UI_ROUTES) >= 8
	for r in UI_ROUTES:
		assert "name" in r and "path" in r and "component" in r


def test_supported_constants_populated():
	assert len(SUPPORTED_CHANGE_TYPES) >= 5
	assert len(SUPPORTED_CAPA_TYPES) >= 3
	assert len(SUPPORTED_DEVIATION_TYPES) >= 4
	assert len(SUPPORTED_DOCUMENT_TYPES) >= 5
	assert len(SUPPORTED_AUDIT_TYPES) >= 4
	assert len(SUPPORTED_VALIDATION_TYPES) >= 4


def test_rules_minimum_count():
	assert len(RULES) >= 20


def test_evaluate_allow_read():
	result = evaluate_capability_rules({"tenant_context_present": True})
	assert result["decision"] == "allow"


def test_evaluate_deny_no_tenant():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "tenant_context_required" for a in result["actions"])


def test_evaluate_deny_change_no_impact():
	result = evaluate_capability_rules({
		"operation": "approve_change",
		"impact_assessed": False,
	})
	assert result["decision"] == "deny"
	assert any("impact" in a["rule"] for a in result["actions"])


def test_evaluate_deny_document_no_approval():
	result = evaluate_capability_rules({
		"operation": "make_document_effective",
		"approved": False,
	})
	assert result["decision"] == "deny"
	assert any("document_approval" in a["rule"] for a in result["actions"])


def test_evaluate_deny_capa_no_root_cause():
	result = evaluate_capability_rules({
		"operation": "close_capa",
		"root_cause_identified": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_critical_deviation_timeline():
	result = evaluate_capability_rules({
		"operation": "raise_deviation",
		"severity": "critical",
		"within_24h": False,
	})
	assert result["decision"] == "deny"


def test_contract_streaming():
	contract = get_capability_contract()
	streaming = contract["streaming"]
	assert streaming["processor"] == "bytewax"
	assert "capa_raised" in streaming["events"]
	assert "deviation_raised" in streaming["events"]


def test_theme_components():
	contract = get_capability_contract()
	comps = contract["theme"]["components"]
	assert "change_control" in comps
	assert "capa" in comps
	assert "deviations" in comps


def test_contract_requires_auth_and_audit():
	contract = get_capability_contract()
	for cap in ["auth", "audl", "mten", "conf"]:
		assert cap in contract["requires"]


def test_rule_effects_complete():
	for rule in RULES:
		assert "name" in rule
		assert "condition" in rule
		assert "effect" in rule
		assert "decision" in rule["effect"]
		assert "reason" in rule["effect"]
