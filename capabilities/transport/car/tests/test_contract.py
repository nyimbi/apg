"""Capability contract tests for transport_car (Cargo Management)."""

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

_cc = _load("_contract_car", "capability_contract.py")
CAPABILITY_ID = _cc.CAPABILITY_ID
SUPPORTED_CARGO_TYPES = _cc.SUPPORTED_CARGO_TYPES
SUPPORTED_DG_CLASSES = _cc.SUPPORTED_DG_CLASSES
SUPPORTED_BOOKING_STATUSES = _cc.SUPPORTED_BOOKING_STATUSES
SUPPORTED_AGENT_RUNTIMES = _cc.SUPPORTED_AGENT_RUNTIMES
SUPPORTED_AGENT_ROLES = _cc.SUPPORTED_AGENT_ROLES

get_capability_contract = _cc.get_capability_contract
evaluate_capability_rules = _cc.evaluate_capability_rules

def test_contract_shape():
	contract = get_capability_contract("tenant-test")
	assert contract["capability"] == "transport_car"
	assert contract["version"] == "1.0.0"
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"] == "6px"
	assert "tenant_id" in contract["configuration_schema"]["required"]
	assert "ui" in contract["configuration_schema"]["required"]
	assert "theme" in contract["configuration_schema"]["required"]


def test_provides_and_requires():
	contract = get_capability_contract("tenant-test")
	assert "cargo_booking_workflow" in contract["provides"]
	assert "dangerous_goods_compliance_workflow" in contract["provides"]
	assert "auth" in contract["requires"]
	assert "audl" in contract["requires"]
	assert "comp" in contract["requires"]


def test_ui_routes():
	contract = get_capability_contract("tenant-test")
	paths = [r["path"] for r in contract["ui"]["routes"]]
	assert "/transport-cargo/dashboard" in paths
	assert "/transport-cargo/bookings" in paths
	assert "/transport-cargo/dangerous-goods" in paths
	assert "/transport-cargo/tracking" in paths
	assert "/transport-cargo/agents" in paths
	assert len(contract["ui"]["routes"]) >= 8


def test_theme_tokens():
	contract = get_capability_contract("tenant-test")
	tokens = contract["theme"]["tokens"]
	for key in ["color.primary", "color.accent", "color.success", "color.warning", "color.danger", "surface.canvas", "surface.panel", "text.primary", "text.secondary", "border.radius", "density"]:
		assert key in tokens


def test_rule_engine_tenant_context():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"


def test_rule_engine_write_without_policy():
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "write", "policy_attached": False})
	assert result["decision"] == "deny"


def test_rule_engine_dg_without_approval():
	result = evaluate_capability_rules({"tenant_context_present": True, "operation": "create_booking", "cargo_type": "hazardous", "dg_approved": False})
	assert result["decision"] == "deny"


def test_rule_engine_batch_non_bytewax():
	result = evaluate_capability_rules({"tenant_context_present": True, "operation": "cargo_batch", "event_stream": "kafka"})
	assert result["decision"] == "deny"


def test_rule_engine_cross_tenant():
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "write", "cross_tenant_access": True})
	assert result["decision"] == "deny"


def test_rule_engine_allow():
	result = evaluate_capability_rules({"tenant_id": "t1", "tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] == "allow"


def test_supported_constants():
	assert len(SUPPORTED_CARGO_TYPES) >= 10
	assert len(SUPPORTED_DG_CLASSES) >= 8
	assert len(SUPPORTED_BOOKING_STATUSES) >= 5
	assert len(SUPPORTED_AGENT_RUNTIMES) >= 3
	assert len(SUPPORTED_AGENT_ROLES) >= 4


def test_streaming_config():
	contract = get_capability_contract("tenant-test")
	streaming = contract["streaming"]
	assert streaming["key"] == "tenant_id"
	assert "cargo_booked" in streaming["events"]
	assert len(streaming["events"]) >= 5
