"""Capability contract tests for transport_tra (Asset Tracking)."""

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

_cc = _load("_contract_tra", "capability_contract.py")
CAPABILITY_ID = _cc.CAPABILITY_ID
SUPPORTED_ASSET_TYPES = _cc.SUPPORTED_ASSET_TYPES
SUPPORTED_GEOFENCE_TYPES = _cc.SUPPORTED_GEOFENCE_TYPES
SUPPORTED_COLD_CHAIN_STANDARDS = _cc.SUPPORTED_COLD_CHAIN_STANDARDS
SUPPORTED_AGENT_RUNTIMES = _cc.SUPPORTED_AGENT_RUNTIMES

get_capability_contract = _cc.get_capability_contract
evaluate_capability_rules = _cc.evaluate_capability_rules

def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "transport_tra"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"


def test_ui_routes():
	c = get_capability_contract("t1")
	paths = [r["path"] for r in c["ui"]["routes"]]
	assert "/transport-tracking/map" in paths
	assert "/transport-tracking/geofencing" in paths
	assert "/transport-tracking/cold-chain" in paths
	assert "/transport-tracking/containers" in paths
	assert "/transport-tracking/agents" in paths
	assert len(c["ui"]["routes"]) >= 8


def test_rule_tamper_escalation():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "update_asset_location", "tamper_detected": True})["decision"] == "deny"


def test_rule_tenant_context():
	assert evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"


def test_rule_batch_non_bytewax():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "tracking_batch", "event_stream": "kinesis"})["decision"] == "deny"


def test_provides():
	c = get_capability_contract("t1")
	assert "realtime_gps_tracking_workflow" in c["provides"]
	assert "cold_chain_monitoring_workflow" in c["provides"]
	assert "container_tracking_workflow" in c["provides"]


def test_supported_constants():
	assert len(SUPPORTED_ASSET_TYPES) >= 8
	assert len(SUPPORTED_GEOFENCE_TYPES) >= 6
	assert len(SUPPORTED_COLD_CHAIN_STANDARDS) >= 5
