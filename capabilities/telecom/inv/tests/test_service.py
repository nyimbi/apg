"""Service-level tests for telecom_inv."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name, path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_describe_returns_contract():
	mod = _load("svc_desc_inv", PACKAGE_DIR / "service.py")
	svc = mod.TelecomInvService()
	c = svc.describe("t1")
	assert c["capability"] == "telecom_inv"


def test_all_asset_types_accepted():
	mod = _load("svc_types_inv", PACKAGE_DIR / "service.py")
	svc = mod.TelecomInvService()
	for i, atype in enumerate(["base_station", "antenna", "router", "switch", "server"]):
		asset = svc.commission_asset(f"a{i}", "t1", atype, f"SN-{i}", "ericsson", "model", "loc", "2026-01-01")
		assert asset["asset_type"] == atype


def test_ip_block_release():
	mod = _load("svc_ip_inv", PACKAGE_DIR / "service.py")
	svc = mod.TelecomInvService()
	block = svc.allocate_ip_block("b1", "t1", "ipv4", "10.10.0.0", 24, "lan_subnet", "VRF-A", "server-1", "2026-01-01")
	assert block["allocated_to"] == "server-1"
	released = svc.release_ip_block("b1", "t1")
	assert released["allocated_to"] is None


def test_circuit_status_transitions():
	mod = _load("svc_cct_inv", PACKAGE_DIR / "service.py")
	svc = mod.TelecomInvService()
	cct = svc.provision_circuit("c1", "t1", "stm1", "site-a", "site-b", "155Mbps", "2026-01-01")
	assert cct["status"] == "provisioned"
	svc.update_circuit_status("c1", "t1", "active")
	assert svc.circuits[("t1", "c1")].status == "active"


def test_topology_domains_recorded():
	mod = _load("svc_topo_inv", PACKAGE_DIR / "service.py")
	svc = mod.TelecomInvService()
	for domain in ["core", "metro", "access"]:
		topo = svc.record_topology(f"topo-{domain}", "t1", "ring", domain, f"{domain.title()} Ring", "desc", "[]", "[]", "2026-01-01")
		assert topo["domain"] == domain


def test_discrepancy_workflow():
	mod = _load("svc_disc_inv", PACKAGE_DIR / "service.py")
	svc = mod.TelecomInvService()
	svc.commission_asset("a1", "t1", "router", "SN-001", "cisco", "ASR9K", "DC1", "2026-01-01")
	disc = svc.record_discrepancy("rec-1", "t1", "a1", "Asset shows as active but not in field audit")
	assert disc["status"] == "open"
	resolved = svc.approve_reconciliation("rec-1", "t1", "approval-ref", "auditor", "2026-01-10")
	assert resolved["status"] == "resolved"
	assert resolved["resolved_by"] == "auditor"


def test_multi_tenant_asset_isolation():
	mod = _load("svc_iso_inv", PACKAGE_DIR / "service.py")
	svc = mod.TelecomInvService()
	svc.commission_asset("a1", "tenant-a", "router", "SN-A", "cisco", "model", "DC-A", "2026-01-01")
	svc.commission_asset("a1", "tenant-b", "switch", "SN-B", "juniper", "model", "DC-B", "2026-01-01")
	assert svc.assets[("tenant-a", "a1")].asset_type == "router"
	assert svc.assets[("tenant-b", "a1")].asset_type == "switch"


def test_decommission_sets_status():
	mod = _load("svc_decom_inv", PACKAGE_DIR / "service.py")
	svc = mod.TelecomInvService()
	svc.commission_asset("a1", "t1", "antenna", "SN-DECOM", "nokia", "AHB-002", "Tower-1", "2026-01-01")
	decom = svc.decommission_asset("a1", "t1", "decom-approval-001")
	assert decom["status"] == "decommissioned"


def test_ipv6_block_allocation():
	mod = _load("svc_ipv6_inv", PACKAGE_DIR / "service.py")
	svc = mod.TelecomInvService()
	block = svc.allocate_ip_block("ipv6-1", "t1", "ipv6", "2001:db8::", 32, "transit", "VRF-IPV6", None, "2026-01-01")
	assert block["ip_version"] == "ipv6"
	assert block["prefix_length"] == 32
