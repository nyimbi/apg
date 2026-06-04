"""Contract shape and rule evaluation tests for APG Electoral & Civil Registration."""

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
	mod = _load("contract_gov_ele", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")
	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "government_ele"
	assert "voter_registration_workflow" in contract["provides"]
	assert "biometric_deduplication_workflow" in contract["provides"]


def test_duplicate_voter_denied():
	mod = _load("rules_gov_ele", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "register_voter", "duplicate_detected": True})
	assert result["decision"] == "deny"


def test_underage_voter_denied():
	mod = _load("rules_gov_ele2", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "register_voter", "of_voting_age": False})
	assert result["decision"] == "deny"


def test_result_manipulation_denied():
	mod = _load("rules_gov_ele3", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "collate_result", "manipulation_detected": True})
	assert result["decision"] == "deny"


def test_ui_routes_include_civil_registry():
	mod = _load("ui_gov_ele", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("t")
	paths = [r["path"] for r in contract["ui"]["routes"]]
	assert "/government-ele/civil-registry" in paths
	assert "/government-ele/deduplication" in paths
