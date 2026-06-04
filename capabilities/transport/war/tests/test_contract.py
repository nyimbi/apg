"""Capability contract tests for transport_war (Warehouse Operations)."""

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

_cc = _load("_contract_war", "capability_contract.py")
CAPABILITY_ID = _cc.CAPABILITY_ID
SUPPORTED_WAREHOUSE_TYPES = _cc.SUPPORTED_WAREHOUSE_TYPES
SUPPORTED_PICK_METHODS = _cc.SUPPORTED_PICK_METHODS
SUPPORTED_CYCLE_COUNT_TYPES = _cc.SUPPORTED_CYCLE_COUNT_TYPES
SUPPORTED_AGENT_RUNTIMES = _cc.SUPPORTED_AGENT_RUNTIMES

get_capability_contract = _cc.get_capability_contract
evaluate_capability_rules = _cc.evaluate_capability_rules

def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "transport_war"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "6px"


def test_ui_routes():
	c = get_capability_contract("t1")
	paths = [r["path"] for r in c["ui"]["routes"]]
	assert "/transport-warehouse/receiving" in paths
	assert "/transport-warehouse/picking" in paths
	assert "/transport-warehouse/cycle-count" in paths
	assert "/transport-warehouse/cross-dock" in paths
	assert "/transport-warehouse/agents" in paths
	assert len(c["ui"]["routes"]) >= 8


def test_rule_unapproved_stock_adjustment():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "adjust_inventory", "approval_present": False})["decision"] == "deny"


def test_rule_inventory_manipulation():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "adjust_inventory", "manipulation_detected": True})["decision"] == "deny"


def test_rule_cold_chain_no_temp_check():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "receive_goods", "cold_chain_required": True, "temperature_checked": False})["decision"] == "deny"


def test_rule_batch_non_bytewax():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "warehouse_batch", "event_stream": "activemq"})["decision"] == "deny"


def test_provides():
	c = get_capability_contract("t1")
	assert "warehouse_receiving_workflow" in c["provides"]
	assert "cross_docking_workflow" in c["provides"]
	assert "wms_integration_workflow" in c["provides"]


def test_supported_constants():
	assert len(SUPPORTED_WAREHOUSE_TYPES) >= 8
	assert len(SUPPORTED_PICK_METHODS) >= 6
	assert len(SUPPORTED_CYCLE_COUNT_TYPES) >= 5
