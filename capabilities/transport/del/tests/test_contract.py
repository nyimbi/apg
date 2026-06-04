"""Capability contract tests for transport_del (Delivery Management)."""

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

_cc = _load("_contract_del", "capability_contract.py")
CAPABILITY_ID = _cc.CAPABILITY_ID
SUPPORTED_DELIVERY_TYPES = _cc.SUPPORTED_DELIVERY_TYPES
SUPPORTED_POD_TYPES = _cc.SUPPORTED_POD_TYPES
SUPPORTED_SLA_TIERS = _cc.SUPPORTED_SLA_TIERS
SUPPORTED_AGENT_RUNTIMES = _cc.SUPPORTED_AGENT_RUNTIMES

get_capability_contract = _cc.get_capability_contract
evaluate_capability_rules = _cc.evaluate_capability_rules

def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "transport_del"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"
	for key in ["tenant_id", "ui", "theme"]:
		assert key in c["configuration_schema"]["required"]


def test_ui_routes():
	c = get_capability_contract("t1")
	paths = [r["path"] for r in c["ui"]["routes"]]
	assert "/transport-delivery/dashboard" in paths
	assert "/transport-delivery/pod" in paths
	assert "/transport-delivery/failed" in paths
	assert "/transport-delivery/sla" in paths
	assert "/transport-delivery/agents" in paths
	assert len(c["ui"]["routes"]) >= 8


def test_rule_tenant_context():
	assert evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"


def test_rule_pod_falsification():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "record_pod", "pod_falsification_detected": True})["decision"] == "deny"


def test_rule_max_reschedule():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "reschedule_delivery", "max_reschedule_exceeded": True})["decision"] == "deny"


def test_rule_batch_non_bytewax():
	assert evaluate_capability_rules({"tenant_context_present": True, "operation": "delivery_batch", "event_stream": "sqs"})["decision"] == "deny"


def test_streaming_events():
	c = get_capability_contract("t1")
	events = c["streaming"]["events"]
	assert "delivery_created" in events
	assert "pod_recorded" in events
	assert "sla_breached" in events


def test_supported_constants():
	assert len(SUPPORTED_DELIVERY_TYPES) >= 8
	assert len(SUPPORTED_POD_TYPES) >= 5
	assert len(SUPPORTED_SLA_TIERS) >= 4
