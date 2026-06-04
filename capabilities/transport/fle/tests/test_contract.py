"""Capability contract tests for transport_fle (Fleet Management)."""

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

_cc = _load("_contract_fle", "capability_contract.py")
CAPABILITY_ID = _cc.CAPABILITY_ID
SUPPORTED_VEHICLE_TYPES = _cc.SUPPORTED_VEHICLE_TYPES
SUPPORTED_DRIVER_STATUSES = _cc.SUPPORTED_DRIVER_STATUSES
SUPPORTED_COMPLIANCE_STANDARDS = _cc.SUPPORTED_COMPLIANCE_STANDARDS
SUPPORTED_AGENT_RUNTIMES = _cc.SUPPORTED_AGENT_RUNTIMES

get_capability_contract = _cc.get_capability_contract
evaluate_capability_rules = _cc.evaluate_capability_rules

def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "transport_fle"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"


def test_ui_routes():
	c = get_capability_contract("t1")
	paths = [r["path"] for r in c["ui"]["routes"]]
	assert "/transport-fleet/vehicles" in paths
	assert "/transport-fleet/drivers" in paths
	assert "/transport-fleet/compliance" in paths
	assert "/transport-fleet/agents" in paths
	assert len(c["ui"]["routes"]) >= 8


def test_rule_non_compliant_vehicle():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "dispatch_vehicle", "compliance_check_passed": False})["decision"] == "deny"


def test_rule_unlicenced_driver():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "assign_driver", "driver_licenced": False})["decision"] == "deny"


def test_rule_tenant_context():
	assert evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"


def test_rule_batch():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "fleet_batch", "event_stream": "non_bytewax"})["decision"] == "deny"


def test_provides():
	c = get_capability_contract("t1")
	assert "vehicle_lifecycle_workflow" in c["provides"]
	assert "fleet_compliance_workflow" in c["provides"]


def test_supported_constants():
	assert len(SUPPORTED_VEHICLE_TYPES) >= 10
	assert len(SUPPORTED_DRIVER_STATUSES) >= 5
	assert len(SUPPORTED_COMPLIANCE_STANDARDS) >= 6
