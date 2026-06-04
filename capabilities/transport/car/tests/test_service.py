"""Service tests for transport_car (Cargo Management)."""

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

_cc = _load("_contract2_car", "capability_contract.py")
import sys as _sys
_sys.modules["capability_contract"] = _cc
_models_mod = _load("_models_car", "models.py")
_sys.modules["models"] = _models_mod
_svc_mod = _load("_service_car", "service.py")
CargoManagementService = _svc_mod.CargoManagementService

def test_booking_lifecycle():
	svc = CargoManagementService()
	booking = svc.create_booking("b1", "tenant-a", "general", "shipper-1", "consignee-1", "NBO", "LHR", 1200.0, 10.0, "fob", "pallet")
	assert booking["id"] == "b1"
	assert booking["status"] == "confirmed"
	assert booking["cargo_type"] == "general"


def test_manifest_requires_booking():
	svc = CargoManagementService()
	with pytest.raises(PermissionError):
		svc.create_manifest("m1", "tenant-a", "missing-booking", "DECL-001")


def test_manifest_created_with_valid_booking():
	svc = CargoManagementService()
	svc.create_booking("b1", "tenant-a", "general", "shipper-1", "consignee-1", "NBO", "LHR", 1200.0, 10.0, "fob", "pallet")
	manifest = svc.create_manifest("m1", "tenant-a", "b1", "DECL-001")
	assert manifest["booking_id"] == "b1"
	assert manifest["status"] == "draft"


def test_dg_declaration():
	svc = CargoManagementService()
	svc.create_booking("b1", "tenant-a", "hazardous", "shipper-1", "consignee-1", "NBO", "LHR", 500.0, 5.0, "cip", "drum")
	dg = svc.declare_dangerous_goods("dg1", "tenant-a", "b1", "class_3_flammable_liquids", "UN1203", "II", "+254700000000", "iata")
	assert dg["dg_class"] == "class_3_flammable_liquids"
	assert dg["un_number"] == "UN1203"


def test_dg_missing_un_number():
	svc = CargoManagementService()
	svc.create_booking("b1", "tenant-a", "hazardous", "s", "c", "NBO", "LHR", 100.0, 1.0, "fob", "drum")
	with pytest.raises(PermissionError, match="un_number_required"):
		svc.declare_dangerous_goods("dg1", "tenant-a", "b1", "class_3_flammable_liquids", "", "II", "+254700000000", "iata")


def test_tracking_event():
	svc = CargoManagementService()
	svc.create_booking("b1", "tenant-a", "general", "s", "c", "NBO", "LHR", 1000.0, 8.0, "fob", "pallet")
	event = svc.update_tracking("e1", "tenant-a", "b1", "in_transit", "Dubai Airport", "2026-06-01T10:00:00Z")
	assert event["event_type"] == "in_transit"
	assert event["location"] == "Dubai Airport"


def test_tracking_unsupported_event():
	svc = CargoManagementService()
	with pytest.raises(PermissionError, match="tracking_event_not_supported"):
		svc.update_tracking("e1", "tenant-a", "b1", "teleported", "Nowhere", "2026-06-01T10:00:00Z")


def test_revenue_record():
	svc = CargoManagementService()
	svc.create_booking("b1", "tenant-a", "general", "s", "c", "NBO", "LHR", 1000.0, 8.0, "fob", "pallet")
	rev = svc.record_revenue("r1", "tenant-a", "b1", "freight_charge", 1500.0, "USD", "INV-001")
	assert rev["amount"] == 1500.0
	assert rev["currency"] == "USD"


def test_revenue_negative_amount():
	svc = CargoManagementService()
	with pytest.raises(PermissionError, match="revenue_amount_must_be_positive"):
		svc.record_revenue("r1", "tenant-a", "b1", "freight_charge", -100.0, "USD", "REF")


def test_cancel_booking():
	svc = CargoManagementService()
	svc.create_booking("b1", "tenant-a", "general", "s", "c", "NBO", "LHR", 1000.0, 8.0, "fob", "pallet")
	result = svc.cancel_booking("b1", "tenant-a")
	assert result["status"] == "cancelled"


def test_tenant_isolation():
	svc = CargoManagementService()
	svc.create_booking("b1", "tenant-a", "general", "s", "c", "NBO", "LHR", 1000.0, 8.0, "fob", "pallet")
	svc.create_booking("b1", "tenant-b", "bulk", "s2", "c2", "NBI", "DXB", 500.0, 4.0, "cfr", "drum")
	assert svc.dashboard_summary("tenant-a")["booking_count"] == 1
	assert svc.dashboard_summary("tenant-b")["booking_count"] == 1


def test_tenant_context_required():
	svc = CargoManagementService()
	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.create_booking("b1", "", "general", "s", "c", "NBO", "LHR", 1000.0, 8.0, "fob", "pallet")


def test_register_agent():
	svc = CargoManagementService()
	agent = svc.register_cargo_agent("a1", "tenant-a", "Cargo Bot", "codex", "booking_agent", "cargo booking scope")
	assert agent["runtime"] == "codex"
	assert agent["role"] == "booking_agent"


def test_batch_requires_bytewax():
	svc = CargoManagementService()
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("tenant-a", 5, event_stream="kafka")


def test_batch_valid():
	svc = CargoManagementService()
	result = svc.validate_batch("tenant-a", 10)
	assert result["processor"] == "bytewax"
	assert result["accepted"] is True


def test_dashboard_summary():
	svc = CargoManagementService()
	svc.create_booking("b1", "t1", "general", "s", "c", "NBO", "LHR", 1000.0, 8.0, "fob", "pallet")
	svc.update_tracking("e1", "t1", "b1", "in_transit", "Dubai", "2026-06-01T10:00:00Z")
	summary = svc.dashboard_summary("t1")
	assert summary["booking_count"] == 1
	assert summary["tracking_event_count"] == 1
	assert summary["audit_event_count"] >= 2
