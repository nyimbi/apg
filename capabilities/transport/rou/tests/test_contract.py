"""Capability contract tests for transport_rou (Route Optimisation)."""

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

_cc = _load("_contract_rou", "capability_contract.py")
CAPABILITY_ID = _cc.CAPABILITY_ID
SUPPORTED_ROUTE_TYPES = _cc.SUPPORTED_ROUTE_TYPES
SUPPORTED_OPTIMISATION_OBJECTIVES = _cc.SUPPORTED_OPTIMISATION_OBJECTIVES
SUPPORTED_REROUTING_TRIGGERS = _cc.SUPPORTED_REROUTING_TRIGGERS
SUPPORTED_AGENT_RUNTIMES = _cc.SUPPORTED_AGENT_RUNTIMES

get_capability_contract = _cc.get_capability_contract
evaluate_capability_rules = _cc.evaluate_capability_rules

def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "transport_rou"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"


def test_ui_routes():
	c = get_capability_contract("t1")
	paths = [r["path"] for r in c["ui"]["routes"]]
	assert "/transport-route/routes" in paths
	assert "/transport-route/optimisation" in paths
	assert "/transport-route/traffic" in paths
	assert "/transport-route/multimodal" in paths
	assert "/transport-route/agents" in paths
	assert len(c["ui"]["routes"]) >= 8


def test_rule_unvalidated_address():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "plan_route", "address_validated": False})["decision"] == "deny"


def test_rule_capacity_violation():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "plan_route", "capacity_constraint_violated": True})["decision"] == "deny"


def test_rule_max_stops():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "plan_route", "stops_exceed_maximum": True})["decision"] == "deny"


def test_rule_batch_non_bytewax():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "route_batch", "event_stream": "nats"})["decision"] == "deny"


def test_provides():
	c = get_capability_contract("t1")
	assert "multi_stop_route_planning_workflow" in c["provides"]
	assert "dynamic_rerouting_workflow" in c["provides"]
	assert "multimodal_routing_workflow" in c["provides"]


def test_supported_constants():
	assert len(SUPPORTED_ROUTE_TYPES) >= 8
	assert len(SUPPORTED_OPTIMISATION_OBJECTIVES) >= 6
	assert len(SUPPORTED_REROUTING_TRIGGERS) >= 6
