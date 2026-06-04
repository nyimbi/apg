"""Tests for telecom_inv capability contract and service."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_contract_shape():
	mod = _load("cc_inv", PACKAGE_DIR / "capability_contract.py")
	c = mod.get_capability_contract("t1")
	assert c["capability"] == "telecom_inv"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"
	assert "asset_inventory_workflow" in c["provides"]
	assert "ipam_workflow" in c["provides"]
	assert len(c["ui"]["routes"]) >= 8
	assert len(c["rule_engine"]["rules"]) >= 20


def test_rule_engine():
	mod = _load("re_inv", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "inv_batch", "event_stream": "kinesis"})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "decommission_asset", "approval_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})["decision"] == "allow"


def test_inventory_lifecycle():
	mod = _load("svc_inv", PACKAGE_DIR / "service.py")
	svc = mod.TelecomInvService()

	site = svc.register_site("site-1", "t1", "Nairobi Site A", "tower", -1.286389, 36.817223, "CBD Nairobi", "Central")
	asset = svc.commission_asset("asset-1", "t1", "base_station", "SN-12345", "ericsson", "RBS 6201", site["id"], "2026-01-01")
	circuit = svc.provision_circuit("cct-1", "t1", "ethernet_10g", "site-1", "site-2", "10Gbps", "2026-01-01")
	ip_block = svc.allocate_ip_block("ip-1", "t1", "ipv4", "10.0.0.0", 24, "lan_subnet", "VRF-DEFAULT", "site-1", "2026-01-01")
	topology = svc.record_topology("topo-1", "t1", "ring", "metro", "Metro Ring", "Nairobi metro ring", '["site-1","site-2"]', '[]', "2026-01-01")
	discrepancy = svc.record_discrepancy("rec-1", "t1", asset["id"], "Asset not found during audit")
	reconciled = svc.approve_reconciliation("rec-1", "t1", "approval-ref", "auditor-1", "2026-01-10")
	decommission = svc.decommission_asset(asset["id"], "t1", "decom-approval-1")
	released = svc.release_ip_block(ip_block["id"], "t1")
	agent = svc.register_agent("agt-1", "t1", "INV Agent", "codex", "inventory_auditor", "inventory operations")
	batch = svc.validate_batch("t1", 5)
	summary = svc.dashboard_summary("t1")

	assert asset["asset_type"] == "base_station"
	assert circuit["circuit_type"] == "ethernet_10g"
	assert ip_block["ip_version"] == "ipv4"
	assert topology["topology_type"] == "ring"
	assert reconciled["status"] == "resolved"
	assert decommission["status"] == "decommissioned"
	assert released["allocated_to"] is None
	assert agent["role"] == "inventory_auditor"
	assert batch["processor"] == "bytewax"
	assert summary["asset_count"] == 1


def test_guardrails():
	mod = _load("svc_guard_inv", PACKAGE_DIR / "service.py")
	svc = mod.TelecomInvService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.commission_asset("a", "", "base_station", "SN", "ericsson", "model", "loc", "2026-01-01")
	with pytest.raises(PermissionError, match="asset_type_not_supported"):
		svc.commission_asset("a", "t1", "submarine", "SN-1", "other", "model", "loc", "2026-01-01")
	with pytest.raises(PermissionError, match="serial_number_required"):
		svc.commission_asset("a", "t1", "router", "", "cisco", "ASR9K", "DC1", "2026-01-01")
	with pytest.raises(PermissionError, match="circuit_type_not_supported"):
		svc.provision_circuit("c", "t1", "carrier_pigeon", "a-end", "z-end", "1bps", "2026-01-01")
	with pytest.raises(PermissionError, match="ip_version_not_supported"):
		svc.allocate_ip_block("b", "t1", "ipv9", "1.2.3.4", 24, "lan_subnet", "VRF-1", None, "2026-01-01")
	with pytest.raises(PermissionError, match="decommission_approval_required"):
		svc.decommission_asset("asset-x", "t1", "")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 1, event_stream="sqs")


def test_api_and_views():
	api = _load("api_inv", PACKAGE_DIR / "api.py")
	views = _load("views_inv", PACKAGE_DIR / "views.py")

	asset = api.commission_asset({"tenant_id": "t-api", "asset_id": "a-api", "asset_type": "router", "serial_number": "SN-API-001", "location": "DC1"})
	circuit = api.provision_circuit({"tenant_id": "t-api", "circuit_id": "c-api", "circuit_type": "ethernet_10g", "a_end": "site-a", "z_end": "site-z", "capacity": "10Gbps"})
	ip_block = api.allocate_ip_block({"tenant_id": "t-api", "block_id": "ip-api", "ip_version": "ipv4", "prefix": "192.168.1.0", "prefix_length": 24, "block_type": "lan_subnet", "vrf": "VRF-MGMT"})
	batch = api.validate_batch({"tenant_id": "t-api", "item_count": 2})
	db = views.dashboard_model(api.service(), "t-api")
	ipam = views.ipam_console_model(api.service(), "t-api")

	assert asset["asset_type"] == "router"
	assert circuit["circuit_type"] == "ethernet_10g"
	assert ip_block["ip_version"] == "ipv4"
	assert batch["processor"] == "bytewax"
	assert db["summary"]["asset_count"] == 1
	assert len(ipam["all_blocks"]) == 1
