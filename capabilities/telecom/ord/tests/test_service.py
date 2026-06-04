"""Service-level tests for telecom_ord."""

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
	mod = _load("svc_desc_ord", PACKAGE_DIR / "service.py")
	svc = mod.TelecomOrdService()
	c = svc.describe("t1")
	assert c["capability"] == "telecom_ord"


def test_all_order_types_accepted():
	mod = _load("svc_types_ord", PACKAGE_DIR / "service.py")
	svc = mod.TelecomOrdService()
	for i, otype in enumerate(["new_service", "change_service", "terminate_service", "sim_swap", "plan_change"]):
		order = svc.submit_order(f"ord-{i}", "t1", otype, f"cust-{i}", "web_self_service", "normal", "2026-01-01")
		assert order["order_type"] == otype


def test_decomposition_requires_validated_order():
	mod = _load("svc_decomp_ord", PACKAGE_DIR / "service.py")
	svc = mod.TelecomOrdService()
	svc.submit_order("ord-1", "t1", "new_service", "cust-1", "web_self_service", "normal", "2026-01-01")
	with pytest.raises(PermissionError, match="order_must_be_valid_for_decomposition"):
		svc.decompose_order("ord-1", "t1")
	svc.validate_order("ord-1", "t1")
	decomposed = svc.decompose_order("ord-1", "t1")
	assert decomposed["status"] == "decomposed"


def test_fallout_retry_increments_count():
	mod = _load("svc_retry_ord", PACKAGE_DIR / "service.py")
	svc = mod.TelecomOrdService()
	svc.submit_order("ord-1", "t1", "new_service", "cust-1", "web_self_service", "normal", "2026-01-01")
	svc.validate_order("ord-1", "t1")
	svc.decompose_order("ord-1", "t1")
	svc.record_fallout("fall-1", "t1", "ord-1", "network_error", "NE timeout")
	svc.retry_fallout("fall-1", "t1")
	svc.retry_fallout("fall-1", "t1")
	fallout = svc.fallouts[("t1", "fall-1")]
	assert fallout.retry_count == 2


def test_all_channels_supported():
	mod = _load("svc_chan_ord", PACKAGE_DIR / "service.py")
	svc = mod.TelecomOrdService()
	for i, channel in enumerate(["retail_store", "web_self_service", "mobile_app", "call_centre"]):
		order = svc.submit_order(f"ord-{i}", "t1", "new_service", f"cust-{i}", channel, "normal", "2026-01-01")
		assert order["channel"] == channel


def test_portability_requires_donor_and_msisdn():
	mod = _load("svc_port_ord", PACKAGE_DIR / "service.py")
	svc = mod.TelecomOrdService()
	svc.submit_order("ord-port", "t1", "number_portability", "cust-port", "call_centre", "high", "2026-01-01")
	with pytest.raises(PermissionError, match="msisdn_required_for_portability"):
		svc.submit_portability_request("port-1", "t1", "ord-port", "", "Safaricom", "Airtel", "2026-01-05")
	port = svc.submit_portability_request("port-ok", "t1", "ord-port", "+254700000001", "Safaricom", "Airtel", "2026-01-05")
	assert port["msisdn"] == "+254700000001"


def test_multi_tenant_order_isolation():
	mod = _load("svc_iso_ord", PACKAGE_DIR / "service.py")
	svc = mod.TelecomOrdService()
	svc.submit_order("ord-1", "tenant-a", "new_service", "cust-a", "web_self_service", "normal", "2026-01-01")
	svc.submit_order("ord-1", "tenant-b", "change_service", "cust-b", "mobile_app", "high", "2026-01-01")
	assert svc.orders[("tenant-a", "ord-1")].order_type == "new_service"
	assert svc.orders[("tenant-b", "ord-1")].order_type == "change_service"


def test_task_dependency_stored():
	mod = _load("svc_task_ord", PACKAGE_DIR / "service.py")
	svc = mod.TelecomOrdService()
	svc.submit_order("ord-1", "t1", "new_service", "cust-1", "web_self_service", "normal", "2026-01-01")
	svc.validate_order("ord-1", "t1")
	svc.decompose_order("ord-1", "t1")
	t1 = svc.create_task("tsk-1", "t1", "ord-1", "customer_verification")
	t2 = svc.create_task("tsk-2", "t1", "ord-1", "sim_provisioning", depends_on="tsk-1")
	assert t1["depends_on"] is None
	assert t2["depends_on"] == "tsk-1"
