"""Contract shape and rule evaluation tests for APG Tax Administration."""

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
	mod = _load("contract_gov_tax", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")
	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "government_tax"
	assert "taxpayer_registration_workflow" in contract["provides"]
	assert "audit_case_management_workflow" in contract["provides"]


def test_duplicate_pin_denied():
	mod = _load("rules_gov_tax", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "register_taxpayer", "tax_type_supported": True, "pin_present": True, "national_id_present": True, "evidence_present": True, "duplicate_pin": True})
	assert result["decision"] == "deny"


def test_objection_outside_deadline_denied():
	mod = _load("rules_gov_tax2", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "file_objection", "assessment_present": True, "grounds_present": True, "within_deadline": False})
	assert result["decision"] == "deny"


def test_debt_collection_without_demand_denied():
	mod = _load("rules_gov_tax3", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "initiate_collection", "assessed_liability_present": True, "demand_notice_issued": False})
	assert result["decision"] == "deny"


def test_ui_routes_include_audits_and_objections():
	mod = _load("ui_gov_tax", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("t")
	paths = [r["path"] for r in contract["ui"]["routes"]]
	assert "/government-tax/audits" in paths
	assert "/government-tax/objections" in paths
	assert "/government-tax/debt-collection" in paths
