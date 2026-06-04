"""Service tests for transport_del (Delivery Management)."""

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

_cc = _load("_contract2_del", "capability_contract.py")
import sys as _sys
_sys.modules["capability_contract"] = _cc
_models_mod = _load("_models_del", "models.py")
_sys.modules["models"] = _models_mod
_svc_mod = _load("_service_del", "service.py")
DeliveryManagementService = _svc_mod.DeliveryManagementService

def test_create_delivery():
	svc = DeliveryManagementService()
	d = svc.create_delivery("d1", "t1", "standard", "John Doe", "123 Main St, Nairobi", "2026-06-01T09:00:00Z", "2026-06-01T12:00:00Z")
	assert d["status"] == "pending"
	assert d["delivery_type"] == "standard"


def test_delivery_missing_address():
	svc = DeliveryManagementService()
	with pytest.raises(PermissionError, match="delivery_address_required"):
		svc.create_delivery("d1", "t1", "standard", "John", "", "2026-06-01T09:00:00Z", "2026-06-01T12:00:00Z")


def test_record_pod():
	svc = DeliveryManagementService()
	svc.create_delivery("d1", "t1", "standard", "John", "123 Main St", "2026-06-01T09:00:00Z", "2026-06-01T12:00:00Z")
	pod = svc.record_pod("p1", "t1", "d1", "signature", "-1.2921,36.8219", "2026-06-01T11:30:00Z", "J. Doe")
	assert pod["pod_type"] == "signature"
	assert pod["geo_stamp"] == "-1.2921,36.8219"


def test_pod_requires_geo_stamp():
	svc = DeliveryManagementService()
	svc.create_delivery("d1", "t1", "standard", "John", "123 Main St", "2026-06-01T09:00:00Z", "2026-06-01T12:00:00Z")
	with pytest.raises(PermissionError, match="geo_stamp_required"):
		svc.record_pod("p1", "t1", "d1", "signature", "", "2026-06-01T11:30:00Z")


def test_failed_delivery():
	svc = DeliveryManagementService()
	svc.create_delivery("d1", "t1", "standard", "John", "123 Main St", "2026-06-01T09:00:00Z", "2026-06-01T12:00:00Z")
	fail = svc.record_failed_delivery("f1", "t1", "d1", "not_home", "2026-06-01T11:00:00Z", "No answer")
	assert fail["failure_reason"] == "not_home"


def test_reschedule_delivery():
	svc = DeliveryManagementService()
	svc.create_delivery("d1", "t1", "standard", "John", "123 Main St", "2026-06-01T09:00:00Z", "2026-06-01T12:00:00Z")
	r = svc.reschedule_delivery("r1", "t1", "d1", "customer_portal", "2026-06-02T09:00:00Z", "2026-06-02T12:00:00Z")
	assert r["reschedule_count"] == 1


def test_max_reschedule_blocked():
	svc = DeliveryManagementService()
	svc.create_delivery("d1", "t1", "standard", "John", "123 Main St", "2026-06-01T09:00:00Z", "2026-06-01T12:00:00Z")
	for i in range(3):
		svc.reschedule_delivery(f"r{i}", "t1", "d1", "customer_portal", f"2026-06-0{i+2}T09:00:00Z", f"2026-06-0{i+2}T12:00:00Z")
	with pytest.raises(PermissionError, match="max_reschedule_count_exceeded"):
		svc.reschedule_delivery("r4", "t1", "d1", "customer_portal", "2026-06-06T09:00:00Z", "2026-06-06T12:00:00Z")


def test_set_sla():
	svc = DeliveryManagementService()
	sla = svc.set_sla("s1", "t1", "d1", "gold", "2026-06-01T08:00:00Z")
	assert sla["sla_tier"] == "gold"


def test_send_notification():
	svc = DeliveryManagementService()
	n = svc.send_notification("n1", "t1", "d1", "sms", "+254712345678", "eta_notification", "2026-06-01T08:00:00Z")
	assert n["channel"] == "sms"


def test_create_return():
	svc = DeliveryManagementService()
	ret = svc.create_return("ret1", "t1", "d1", "customer_request", "RMA-001", "2026-06-01T14:00:00Z")
	assert ret["rma_number"] == "RMA-001"


def test_tenant_isolation():
	svc = DeliveryManagementService()
	svc.create_delivery("d1", "t1", "standard", "A", "Addr A", "2026-06-01T09:00:00Z", "2026-06-01T12:00:00Z")
	svc.create_delivery("d1", "t2", "express", "B", "Addr B", "2026-06-01T09:00:00Z", "2026-06-01T12:00:00Z")
	assert svc.dashboard_summary("t1")["delivery_count"] == 1
	assert svc.dashboard_summary("t2")["delivery_count"] == 1


def test_register_agent():
	svc = DeliveryManagementService()
	a = svc.register_delivery_agent("a1", "t1", "Delivery Bot", "claude_code", "sla_monitor", "sla scope")
	assert a["role"] == "sla_monitor"


def test_batch_requires_bytewax():
	svc = DeliveryManagementService()
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 5, event_stream="pubsub")
