"""Capability contract tests for transport_mai (Vehicle Maintenance)."""

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

_cc = _load("_contract_mai", "capability_contract.py")
CAPABILITY_ID = _cc.CAPABILITY_ID
SUPPORTED_MAINTENANCE_TYPES = _cc.SUPPORTED_MAINTENANCE_TYPES
SUPPORTED_PARTS_CATEGORIES = _cc.SUPPORTED_PARTS_CATEGORIES
SUPPORTED_ROADWORTHINESS_STANDARDS = _cc.SUPPORTED_ROADWORTHINESS_STANDARDS
SUPPORTED_AGENT_RUNTIMES = _cc.SUPPORTED_AGENT_RUNTIMES

get_capability_contract = _cc.get_capability_contract
evaluate_capability_rules = _cc.evaluate_capability_rules

def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "transport_mai"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"


def test_ui_routes():
	c = get_capability_contract("t1")
	paths = [r["path"] for r in c["ui"]["routes"]]
	assert "/transport-maintenance/jobs" in paths
	assert "/transport-maintenance/parts" in paths
	assert "/transport-maintenance/inspections" in paths
	assert "/transport-maintenance/roadworthiness" in paths
	assert len(c["ui"]["routes"]) >= 8


def test_rule_expired_mot():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "dispatch_vehicle", "mot_expired": True})["decision"] == "deny"


def test_rule_unsafe_vehicle():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "dispatch_vehicle", "vehicle_safe": False})["decision"] == "deny"


def test_rule_no_digital_signature():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "conduct_inspection", "digital_signature_present": False})["decision"] == "deny"


def test_rule_batch_non_bytewax():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "maintenance_batch", "event_stream": "amqp"})["decision"] == "deny"


def test_provides():
	c = get_capability_contract("t1")
	assert "preventive_maintenance_schedule_workflow" in c["provides"]
	assert "roadworthiness_compliance_workflow" in c["provides"]


def test_supported_constants():
	assert len(SUPPORTED_MAINTENANCE_TYPES) >= 8
	assert len(SUPPORTED_PARTS_CATEGORIES) >= 10
	assert len(SUPPORTED_ROADWORTHINESS_STANDARDS) >= 5
