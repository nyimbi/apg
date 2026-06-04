"""Capability contract tests for transport_dis (Dispatch Operations)."""

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

_cc = _load("_contract_dis", "capability_contract.py")
CAPABILITY_ID = _cc.CAPABILITY_ID
SUPPORTED_LOAD_TYPES = _cc.SUPPORTED_LOAD_TYPES
SUPPORTED_EXCEPTION_TYPES = _cc.SUPPORTED_EXCEPTION_TYPES
SUPPORTED_OPTIMISATION_MODES = _cc.SUPPORTED_OPTIMISATION_MODES
SUPPORTED_AGENT_RUNTIMES = _cc.SUPPORTED_AGENT_RUNTIMES

get_capability_contract = _cc.get_capability_contract
evaluate_capability_rules = _cc.evaluate_capability_rules

def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "transport_dis"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "6px"


def test_ui_routes():
	c = get_capability_contract("t1")
	paths = [r["path"] for r in c["ui"]["routes"]]
	assert "/transport-dispatch/dashboard" in paths
	assert "/transport-dispatch/board" in paths
	assert "/transport-dispatch/exceptions" in paths
	assert "/transport-dispatch/agents" in paths
	assert len(c["ui"]["routes"]) >= 8


def test_rule_overload_denied():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "plan_load", "load_exceeds_legal_limit": True})["decision"] == "deny"


def test_rule_unlicenced_driver():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "assign_driver", "driver_licenced": False})["decision"] == "deny"


def test_rule_hours_exceeded():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "assign_driver", "hours_of_service_compliant": False})["decision"] == "deny"


def test_rule_batch_non_bytewax():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "dispatch_batch", "event_stream": "rabbitmq"})["decision"] == "deny"


def test_provides():
	c = get_capability_contract("t1")
	assert "load_planning_workflow" in c["provides"]
	assert "exception_management_workflow" in c["provides"]


def test_supported_constants():
	assert len(SUPPORTED_LOAD_TYPES) >= 7
	assert len(SUPPORTED_EXCEPTION_TYPES) >= 6
	assert len(SUPPORTED_OPTIMISATION_MODES) >= 5
