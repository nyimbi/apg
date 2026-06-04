"""Contract shape and rule evaluation tests for APG Case Management."""

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
	mod = _load("contract_gov_cas", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")
	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "government_cas"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "case_agent_workflow" in contract["provides"]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_rule_engine_blocks_missing_tenant():
	mod = _load("rules_gov_cas", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"


def test_rule_engine_blocks_cross_tenant():
	mod = _load("rules_gov_cas2", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "open_case", "cross_tenant": True})
	assert result["decision"] == "deny"


def test_rule_engine_blocks_outcome_without_approval():
	mod = _load("rules_gov_cas3", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "record_outcome", "case_present": True, "outcome_type_supported": True, "approval_present": False, "evidence_present": True})
	assert result["decision"] == "deny"


def test_ui_routes_structure():
	mod = _load("ui_gov_cas", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("t")
	paths = [r["path"] for r in contract["ui"]["routes"]]
	assert "/government-cas/cases" in paths
	assert "/government-cas/sla" in paths


def test_configuration_schema_valid():
	mod = _load("schema_gov_cas", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("t")
	assert "tenant_id" in contract["configuration_schema"]["required"]
