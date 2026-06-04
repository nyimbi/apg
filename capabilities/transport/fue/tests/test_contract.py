"""Capability contract tests for transport_fue (Fuel Management)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load(mod_name: str, filename: str):
	path = PACKAGE_DIR / filename
	spec = importlib.util.spec_from_file_location(mod_name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[mod_name] = mod
	spec.loader.exec_module(mod)
	return mod

_cc = _load("_contract_fue", "capability_contract.py")
CAPABILITY_ID = _cc.CAPABILITY_ID
SUPPORTED_FUEL_TYPES = _cc.SUPPORTED_FUEL_TYPES
SUPPORTED_CARD_PROVIDERS = _cc.SUPPORTED_CARD_PROVIDERS
SUPPORTED_CARBON_STANDARDS = _cc.SUPPORTED_CARBON_STANDARDS
SUPPORTED_AGENT_RUNTIMES = _cc.SUPPORTED_AGENT_RUNTIMES

get_capability_contract = _cc.get_capability_contract
evaluate_capability_rules = _cc.evaluate_capability_rules

def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "transport_fue"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "6px"
	for key in ["tenant_id", "ui", "theme"]:
		assert key in c["configuration_schema"]["required"]


def test_ui_routes():
	c = get_capability_contract("t1")
	paths = [r["path"] for r in c["ui"]["routes"]]
	assert "/transport-fuel/dashboard" in paths
	assert "/transport-fuel/transactions" in paths
	assert "/transport-fuel/cards" in paths
	assert "/transport-fuel/carbon" in paths
	assert len(c["ui"]["routes"]) >= 8


def test_rule_phantom_fill():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "record_transaction", "phantom_fill_detected": True})["decision"] == "deny"


def test_rule_theft_detection():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "record_transaction", "theft_pattern_detected": True})["decision"] == "deny"


def test_rule_negative_quantity():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "record_transaction", "quantity_positive": False})["decision"] == "deny"


def test_rule_batch_non_bytewax():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "fuel_batch", "event_stream": "redis"})["decision"] == "deny"


def test_provides():
	c = get_capability_contract("t1")
	assert "fuel_procurement_workflow" in c["provides"]
	assert "carbon_footprint_reporting_workflow" in c["provides"]


def test_supported_constants():
	assert len(SUPPORTED_FUEL_TYPES) >= 8
	assert len(SUPPORTED_CARD_PROVIDERS) >= 8
	assert len(SUPPORTED_CARBON_STANDARDS) >= 5
