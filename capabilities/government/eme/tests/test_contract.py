"""Contract shape and rule evaluation tests for APG Emergency Management."""

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
	mod = _load("contract_gov_eme", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")
	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "government_eme"
	assert "incident_command_workflow" in contract["provides"]
	assert "after_action_review_workflow" in contract["provides"]


def test_unauthorised_eoc_activation_denied():
	mod = _load("rules_gov_eme", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "update_eoc", "eoc_status_supported": True, "authorised": False})
	assert result["decision"] == "deny"


def test_resource_over_allocation_denied():
	mod = _load("rules_gov_eme2", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "mobilise_resource", "over_allocated": True})
	assert result["decision"] == "deny"


def test_missing_incident_commander_denied():
	mod = _load("rules_gov_eme3", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "declare_incident", "incident_type_supported": True, "severity_supported": True, "location_present": True, "commander_present": False, "evidence_present": True})
	assert result["decision"] == "deny"


def test_ui_routes_include_aar():
	mod = _load("ui_gov_eme", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("t")
	paths = [r["path"] for r in contract["ui"]["routes"]]
	assert "/government-eme/after-action" in paths
	assert "/government-eme/eoc" in paths
