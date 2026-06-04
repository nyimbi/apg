"""Contract shape and rule evaluation tests for APG Budget Management."""

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
	mod = _load("contract_gov_bud", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("tenant-test")
	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "government_bud"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "budget_agent_workflow" in contract["provides"]
	assert contract["theme"]["tokens"]["border.radius"] == "6px"


def test_rule_engine_blocks_missing_tenant_context():
	mod = _load("rules_gov_bud", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"


def test_rule_engine_blocks_commitment_without_balance():
	mod = _load("rules_gov_bud2", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({
		"tenant_id": "t", "tenant_context_present": True,
		"operation": "record_commitment", "negative_balance": True,
	})
	assert result["decision"] == "deny"
	assert any("negative_vote_balance" in a.get("reason", "") for a in result["actions"])


def test_rule_engine_blocks_non_bytewax_batch():
	mod = _load("rules_gov_bud3", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({
		"tenant_id": "t", "tenant_context_present": True,
		"operation": "budget_batch", "event_stream": "queue",
	})
	assert result["decision"] == "deny"


def test_rule_engine_allows_valid_context():
	mod = _load("rules_gov_bud4", PACKAGE_DIR / "capability_contract.py")
	result = mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})
	assert result["decision"] == "allow"


def test_ui_routes_have_required_keys():
	mod = _load("ui_gov_bud", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("t")
	for route in contract["ui"]["routes"]:
		assert route["path"].startswith("/")
		assert route["permission"]
		assert route["component"]


def test_configuration_schema_contains_required_keys():
	mod = _load("schema_gov_bud", PACKAGE_DIR / "capability_contract.py")
	contract = mod.get_capability_contract("t")
	required = contract["configuration_schema"]["required"]
	assert "tenant_id" in required
	assert "ui" in required
	assert "theme" in required
