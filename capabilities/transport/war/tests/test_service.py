"""Service tests for transport_war (Warehouse Operations)."""

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

_cc = _load("_contract2_war", "capability_contract.py")
import sys as _sys
_sys.modules["capability_contract"] = _cc
_models_mod = _load("_models_war", "models.py")
_sys.modules["models"] = _models_mod
_svc_mod = _load("_service_war", "service.py")
WarehouseOperationsService = _svc_mod.WarehouseOperationsService

def test_register_warehouse():
	svc = WarehouseOperationsService()
	w = svc.register_warehouse("wh1", "t1", "distribution_centre", "Nairobi DC", "Industrial Area", "ambient", 5000.0, 8)
	assert w["name"] == "Nairobi DC"
	assert w["dock_door_count"] == 8


def test_receive_goods():
	svc = WarehouseOperationsService()
	r = svc.receive_goods("rc1", "t1", "wh1", "po_based", "supp-1", "PO-001", 10, "2026-06-01T08:00:00Z")
	assert r["receipt_method"] == "po_based"
	assert r["damage_inspection_completed"] is True


def test_receive_cold_chain_without_temp_check():
	svc = WarehouseOperationsService()
	with pytest.raises(PermissionError, match="temperature_check_required_for_cold_chain"):
		svc.receive_goods("rc1", "t1", "wh1", "po_based", "supp-1", "PO-001", 5, "2026-06-01T08:00:00Z", cold_chain_required=True, temperature_checked=False)


def test_execute_putaway():
	svc = WarehouseOperationsService()
	svc.receive_goods("rc1", "t1", "wh1", "po_based", "supp-1", "PO-001", 10, "2026-06-01T08:00:00Z")
	p = svc.execute_putaway("pt1", "t1", "rc1", "zone_based", "A1-001", "op-1")
	assert p["slot_id"] == "A1-001"
	assert p["confirmed"] is True


def test_create_pick_task():
	svc = WarehouseOperationsService()
	pt = svc.create_pick_task("pk1", "t1", "ord-1", "single_order", "wh1", 5, "high", "op-1")
	assert pt["pick_method"] == "single_order"
	assert pt["completed_at"] is None


def test_complete_pick_task():
	svc = WarehouseOperationsService()
	svc.create_pick_task("pk1", "t1", "ord-1", "single_order", "wh1", 5, "high", "op-1")
	pt = svc.complete_pick_task("pk1", "t1", "2026-06-01T10:30:00Z")
	assert pt["completed_at"] == "2026-06-01T10:30:00Z"


def test_create_and_complete_pack():
	svc = WarehouseOperationsService()
	svc.create_pick_task("pk1", "t1", "ord-1", "single_order", "wh1", 5, "medium", "op-1")
	svc.complete_pick_task("pk1", "t1", "2026-06-01T10:30:00Z")
	p = svc.create_pack_task("pac1", "t1", "pk1", "standard_carton", 12.5)
	p2 = svc.complete_packing("pac1", "t1", "2026-06-01T11:00:00Z")
	assert p2["label_printed"] is True
	assert p2["packing_slip_printed"] is True


def test_initiate_cycle_count():
	svc = WarehouseOperationsService()
	cc = svc.initiate_cycle_count("cc1", "t1", "wh1", "abc_analysis", "2026-06-01T07:00:00Z")
	assert cc["count_type"] == "abc_analysis"
	assert cc["approved"] is False


def test_complete_cycle_count():
	svc = WarehouseOperationsService()
	svc.initiate_cycle_count("cc1", "t1", "wh1", "abc_analysis", "2026-06-01T07:00:00Z")
	cc = svc.complete_cycle_count("cc1", "t1", "2026-06-01T12:00:00Z", 0.5, "supervisor-1")
	assert cc["approved"] is True
	assert cc["discrepancy_pct"] == 0.5


def test_adjust_inventory_requires_approval():
	svc = WarehouseOperationsService()
	with pytest.raises(PermissionError, match="unapproved_stock_adjustment_denied"):
		svc.adjust_inventory("adj1", "t1", "wh1", "SKU-001", 100, 95, "damage", "", "2026-06-01T12:00:00Z")


def test_adjust_inventory_approved():
	svc = WarehouseOperationsService()
	adj = svc.adjust_inventory("adj1", "t1", "wh1", "SKU-001", 100, 95, "damage", "supervisor-1", "2026-06-01T12:00:00Z")
	assert adj["quantity_after"] == 95


def test_tenant_isolation():
	svc = WarehouseOperationsService()
	svc.register_warehouse("wh1", "t1", "ambient", "WH A", "Loc A", "ambient", 1000.0, 4)
	svc.register_warehouse("wh1", "t2", "cold_store", "WH B", "Loc B", "chilled_2_8", 500.0, 2)
	assert svc.dashboard_summary("t1")["warehouse_count"] == 1
	assert svc.dashboard_summary("t2")["warehouse_count"] == 1


def test_register_agent():
	svc = WarehouseOperationsService()
	a = svc.register_warehouse_agent("a1", "t1", "Warehouse Bot", "codex", "receiving_agent", "receiving scope")
	assert a["role"] == "receiving_agent"
