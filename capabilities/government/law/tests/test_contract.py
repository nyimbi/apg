"""Contract shape and rule evaluation tests for APG Law Enforcement & Justice."""

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
	mod = _load("contract_gov_law", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")
	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "government_law"
	assert "evidence_chain_of_custody_workflow" in contract["provides"]
	assert "docket_management_workflow" in contract["provides"]


def test_chain_of_custody_breach_denied():
	mod = _load("rules_gov_law", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "record_custody_action", "custody_action_supported": True, "chain_intact": False})
	assert result["decision"] == "deny"


def test_prosecution_without_dpp_reference_denied():
	mod = _load("rules_gov_law2", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "record_prosecution", "dpp_reference_present": False})
	assert result["decision"] == "deny"


def test_ob_number_required():
	mod = _load("rules_gov_law3", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "report_incident", "incident_type_supported": True, "ob_number_present": False})
	assert result["decision"] == "deny"


def test_ui_routes_include_evidence_and_prosecution():
	mod = _load("ui_gov_law", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("t")
	paths = [r["path"] for r in contract["ui"]["routes"]]
	assert "/government-law/evidence" in paths
	assert "/government-law/prosecution" in paths
