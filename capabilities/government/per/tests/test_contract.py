"""Contract shape and rule evaluation tests for APG Permits Management."""

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
	mod = _load("contract_gov_per", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")
	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "government_per"
	assert "permit_application_workflow" in contract["provides"]
	assert "permit_compliance_monitoring_workflow" in contract["provides"]


def test_construction_before_permit_denied():
	mod = _load("rules_gov_per", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "record_commencement", "permit_active": False})
	assert result["decision"] == "deny"


def test_occupation_before_final_inspection_denied():
	mod = _load("rules_gov_per2", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "grant_occupation", "final_inspection_passed": False})
	assert result["decision"] == "deny"


def test_duplicate_permit_denied():
	mod = _load("rules_gov_per3", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "issue_permit", "approved_application_present": True, "permit_number_present": True, "expiry_date_present": True, "duplicate_detected": True})
	assert result["decision"] == "deny"


def test_ui_routes_include_compliance_and_enforcement():
	mod = _load("ui_gov_per", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("t")
	paths = [r["path"] for r in contract["ui"]["routes"]]
	assert "/government-per/compliance" in paths
	assert "/government-per/enforcement" in paths
