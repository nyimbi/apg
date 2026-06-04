"""Tests for pharma_mfg capability contract."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from capabilities.pharma.mfg.capability_contract import (
	CAPABILITY_ID, RULES, SUPPORTED_BATCH_STATUSES, SUPPORTED_DEVIATION_TYPES,
	SUPPORTED_EQUIPMENT_STATUSES, SUPPORTED_GMP_FRAMEWORKS, SUPPORTED_LINE_STATUSES,
	SUPPORTED_MANUFACTURING_TYPES, SUPPORTED_MATERIAL_STATUSES, SUPPORTED_QUALIFICATION_TYPES,
	SUPPORTED_YIELD_TYPES, UI_ROUTES, evaluate_capability_rules, get_capability_contract,
)


def test_contract_shape():
	contract = get_capability_contract("mfg_tenant")
	assert contract["capability"] == CAPABILITY_ID
	assert contract["display_name"] == "Pharmaceutical Manufacturing"
	assert contract["configuration"]["tenant_id"] == "mfg_tenant"


def test_theme_tokens_complete():
	tokens = get_capability_contract()["theme"]["tokens"]
	for k in ["color.primary", "color.accent", "border.radius", "density"]:
		assert k in tokens


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	names = [r["name"] for r in UI_ROUTES]
	assert "batches" in names
	assert "equipment" in names
	assert "deviations" in names


def test_supported_constants():
	assert len(SUPPORTED_BATCH_STATUSES) >= 8
	assert len(SUPPORTED_MANUFACTURING_TYPES) >= 5
	assert len(SUPPORTED_EQUIPMENT_STATUSES) >= 4
	assert len(SUPPORTED_QUALIFICATION_TYPES) >= 4
	assert len(SUPPORTED_DEVIATION_TYPES) >= 4
	assert len(SUPPORTED_YIELD_TYPES) >= 4
	assert len(SUPPORTED_LINE_STATUSES) >= 5
	assert len(SUPPORTED_MATERIAL_STATUSES) >= 4
	assert len(SUPPORTED_GMP_FRAMEWORKS) >= 5


def test_rules_count():
	assert len(RULES) >= 20


def test_evaluate_allow():
	result = evaluate_capability_rules({"tenant_context_present": True})
	assert result["decision"] == "allow"


def test_evaluate_deny_no_master_formula():
	result = evaluate_capability_rules({
		"operation": "create_batch", "master_formula_present": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_qp_release():
	result = evaluate_capability_rules({
		"operation": "release_batch", "qp_signed": False,
	})
	assert result["decision"] == "deny"
	assert any("qp" in a["rule"] for a in result["actions"])


def test_evaluate_deny_line_clearance():
	result = evaluate_capability_rules({
		"operation": "start_batch", "line_cleared": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_equipment_not_qualified():
	result = evaluate_capability_rules({
		"operation": "use_equipment", "equipment_qualified": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_deviation_investigation():
	result = evaluate_capability_rules({
		"operation": "close_deviation", "investigation_completed": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_critical_deviation_timeline():
	result = evaluate_capability_rules({
		"operation": "raise_deviation", "severity": "critical", "within_24h": False,
	})
	assert result["decision"] == "deny"


def test_streaming():
	streaming = get_capability_contract()["streaming"]
	assert "batch_released" in streaming["events"]
	assert "equipment_qualified" in streaming["events"]
	assert "gmp_compliance_required" in streaming["guardrails"]


def test_requires_monitoring():
	assert "moni" in get_capability_contract()["requires"]
