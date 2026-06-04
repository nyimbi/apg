"""Service tests for transport_fue (Fuel Management)."""

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

_cc = _load("_contract2_fue", "capability_contract.py")
import sys as _sys
_sys.modules["capability_contract"] = _cc
_models_mod = _load("_models_fue", "models.py")
_sys.modules["models"] = _models_mod
_svc_mod = _load("_service_fue", "service.py")
FuelManagementService = _svc_mod.FuelManagementService

def test_create_procurement():
	svc = FuelManagementService()
	p = svc.create_procurement("p1", "t1", "bulk_storage", "supp-1", "diesel", 10000.0, 1.45, "KES", "PO-001")
	assert p["fuel_type"] == "diesel"
	assert p["quantity_litres"] == 10000.0


def test_record_transaction():
	svc = FuelManagementService()
	t = svc.record_transaction("tx1", "t1", "fill_up", "v1", "dr1", "diesel", 120.5, 45000.0, 1.45, "KES", "2026-06-01T08:00:00Z")
	assert t["quantity_litres"] == 120.5
	assert t["transaction_type"] == "fill_up"


def test_transaction_missing_vehicle():
	svc = FuelManagementService()
	with pytest.raises(PermissionError, match="vehicle_required"):
		svc.record_transaction("tx1", "t1", "fill_up", "", "dr1", "diesel", 120.5, 45000.0, 1.45, "KES", "2026-06-01T08:00:00Z")


def test_transaction_zero_quantity():
	svc = FuelManagementService()
	with pytest.raises(PermissionError, match="fuel_quantity_must_be_positive"):
		svc.record_transaction("tx1", "t1", "fill_up", "v1", "dr1", "diesel", 0.0, 45000.0, 1.45, "KES", "2026-06-01T08:00:00Z")


def test_phantom_fill_blocked():
	svc = FuelManagementService()
	with pytest.raises(PermissionError, match="phantom_fill_detected"):
		svc.record_transaction("tx1", "t1", "fill_up", "v1", "dr1", "diesel", 120.5, 45000.0, 1.45, "KES", "2026-06-01T08:00:00Z", phantom_fill_detected=True)


def test_register_fuel_card():
	svc = FuelManagementService()
	c = svc.register_fuel_card("fc1", "t1", "shell", "****1234", "v1")
	assert c["provider"] == "shell"
	assert c["active"] is True


def test_fuel_card_invalid_provider():
	svc = FuelManagementService()
	with pytest.raises(PermissionError, match="card_provider_not_supported"):
		svc.register_fuel_card("fc1", "t1", "unknown_oil", "****9999")


def test_reconcile_fuel_card():
	svc = FuelManagementService()
	svc.register_fuel_card("fc1", "t1", "shell", "****1234")
	r = svc.reconcile_fuel_card("rec1", "t1", "fc1", "2026-06-01", "2026-06-30", 5000.0, 5000.0, "KES")
	assert r["reconciled"] is True
	assert r["discrepancy"] == 0.0


def test_record_carbon_emission():
	svc = FuelManagementService()
	r = svc.record_carbon_emission("c1", "t1", "v1", "ghg_protocol", "diesel", 120.5, 320.0, "2026-06-01", "2026-06-30")
	assert r["co2_kg"] == 320.0


def test_register_storage_tank():
	svc = FuelManagementService()
	t = svc.register_storage_tank("tk1", "t1", "above_ground_tank", "Depot A", 50000.0, "diesel", "2026-01-15")
	assert t["capacity_litres"] == 50000.0


def test_tenant_isolation():
	svc = FuelManagementService()
	svc.record_transaction("tx1", "t1", "fill_up", "v1", "dr1", "diesel", 100.0, 45000.0, 1.45, "KES", "2026-06-01T08:00:00Z")
	svc.record_transaction("tx1", "t2", "fill_up", "v2", "dr2", "petrol", 80.0, 22000.0, 1.60, "KES", "2026-06-01T09:00:00Z")
	assert svc.dashboard_summary("t1")["transaction_count"] == 1
	assert svc.dashboard_summary("t2")["transaction_count"] == 1


def test_register_agent():
	svc = FuelManagementService()
	a = svc.register_fuel_agent("a1", "t1", "Fuel Bot", "codex", "consumption_analyst", "fuel analytics")
	assert a["role"] == "consumption_analyst"


def test_batch_valid():
	svc = FuelManagementService()
	r = svc.validate_batch("t1", 50)
	assert r["processor"] == "bytewax"
