"""Contract shape and rule evaluation tests for APG Government Contracts & Procurement."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from capabilities.capability_contract_registry import validate_contract_shape

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_contract_shape_is_valid():
	mod = _load("contract_gov_con", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")
	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "government_con"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "ppda_compliance_workflow" in contract["provides"]


def test_debarred_bidder_denied():
	mod = _load("rules_gov_con", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "record_evaluation", "bidder_debarred": True})
	assert result["decision"] == "deny"


def test_award_without_evaluation_denied():
	mod = _load("rules_gov_con2", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "record_award", "approved_evaluation_present": False, "ppda_notification_present": True, "evidence_present": True})
	assert result["decision"] == "deny"


def test_single_source_requires_justification():
	mod = _load("rules_gov_con3", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "publish_tender", "procurement_method": "direct_procurement", "justification_present": False, "procurement_method_supported": True, "ppda_threshold_present": True, "approver_present": True, "evidence_present": True})
	assert result["decision"] == "deny"


def test_ui_routes_include_ppda():
	mod = _load("ui_gov_con", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("t")
	paths = [r["path"] for r in contract["ui"]["routes"]]
	assert "/government-con/ppda" in paths
	assert "/government-con/debarment" in paths
