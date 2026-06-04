"""Capability contract tests for transport_sch (Transport Scheduling)."""

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

_cc = _load("_contract_sch", "capability_contract.py")
CAPABILITY_ID = _cc.CAPABILITY_ID
SUPPORTED_SCHEDULE_TYPES = _cc.SUPPORTED_SCHEDULE_TYPES
SUPPORTED_CHARTER_TYPES = _cc.SUPPORTED_CHARTER_TYPES
SUPPORTED_CONFLICT_TYPES = _cc.SUPPORTED_CONFLICT_TYPES
SUPPORTED_AGENT_RUNTIMES = _cc.SUPPORTED_AGENT_RUNTIMES

get_capability_contract = _cc.get_capability_contract
evaluate_capability_rules = _cc.evaluate_capability_rules

def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "transport_sch"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"


def test_ui_routes():
	c = get_capability_contract("t1")
	paths = [r["path"] for r in c["ui"]["routes"]]
	assert "/transport-scheduling/schedules" in paths
	assert "/transport-scheduling/shifts" in paths
	assert "/transport-scheduling/charters" in paths
	assert "/transport-scheduling/conflicts" in paths
	assert "/transport-scheduling/agents" in paths
	assert len(c["ui"]["routes"]) >= 8


def test_rule_double_booking():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "assign_resource", "double_booking_detected": True})["decision"] == "deny"


def test_rule_driver_hours_breach():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "create_shift", "driver_hours_compliant": False})["decision"] == "deny"


def test_rule_publish_with_conflicts():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_schedule", "unresolved_conflicts_present": True})["decision"] == "deny"


def test_rule_batch_non_bytewax():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "scheduling_batch", "event_stream": "zeromq"})["decision"] == "deny"


def test_provides():
	c = get_capability_contract("t1")
	assert "load_scheduling_workflow" in c["provides"]
	assert "charter_management_workflow" in c["provides"]


def test_supported_constants():
	assert len(SUPPORTED_SCHEDULE_TYPES) >= 6
	assert len(SUPPORTED_CHARTER_TYPES) >= 5
	assert len(SUPPORTED_CONFLICT_TYPES) >= 5
