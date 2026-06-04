"""Process-local API helpers for APG Warehouse Operations."""

from __future__ import annotations

try:
	from .service import WarehouseOperationsService
except ImportError:
	from service import WarehouseOperationsService  # type: ignore

_SERVICE = WarehouseOperationsService()


def service() -> WarehouseOperationsService:
	return _SERVICE


def register_warehouse(payload: dict):
	return _SERVICE.register_warehouse(payload["warehouse_id"], payload.get("tenant_id", "default"), payload["warehouse_type"], payload["name"], payload["location"], payload.get("storage_condition", "ambient"), payload.get("capacity_sqm", 1000.0), payload.get("dock_door_count", 4), payload.get("policy_attached", True))


def receive_goods(payload: dict):
	return _SERVICE.receive_goods(payload["receipt_id"], payload.get("tenant_id", "default"), payload["warehouse_id"], payload["receipt_method"], payload["supplier_id"], payload.get("po_reference", ""), payload.get("line_count", 1), payload["received_at"], payload.get("barcode_scanned", True), payload.get("damage_inspection_completed", True), payload.get("cold_chain_required", False), payload.get("temperature_checked", False))


def execute_putaway(payload: dict):
	return _SERVICE.execute_putaway(payload["task_id"], payload.get("tenant_id", "default"), payload["receipt_id"], payload.get("strategy", "zone_based"), payload["slot_id"], payload.get("operator_id", ""), payload.get("slot_verified", True))


def create_pick_task(payload: dict):
	return _SERVICE.create_pick_task(payload["task_id"], payload.get("tenant_id", "default"), payload["order_id"], payload.get("pick_method", "single_order"), payload["warehouse_id"], payload.get("lines_count", 1), payload.get("priority", "medium"), payload.get("operator_id", ""))


def complete_pick_task(payload: dict):
	return _SERVICE.complete_pick_task(payload["task_id"], payload.get("tenant_id", "default"), payload["completed_at"])


def create_pack_task(payload: dict):
	return _SERVICE.create_pack_task(payload["task_id"], payload.get("tenant_id", "default"), payload["pick_task_id"], payload["pack_type"], payload["weight_kg"], payload.get("weight_checked", True))


def complete_packing(payload: dict):
	return _SERVICE.complete_packing(payload["task_id"], payload.get("tenant_id", "default"), payload["completed_at"], payload.get("weight_checked", True))


def initiate_cycle_count(payload: dict):
	return _SERVICE.initiate_cycle_count(payload["count_id"], payload.get("tenant_id", "default"), payload["warehouse_id"], payload.get("count_type", "abc_analysis"), payload["initiated_at"])


def complete_cycle_count(payload: dict):
	return _SERVICE.complete_cycle_count(payload["count_id"], payload.get("tenant_id", "default"), payload["completed_at"], payload.get("discrepancy_pct", 0.0), payload["approved_by"])


def adjust_inventory(payload: dict):
	return _SERVICE.adjust_inventory(payload["adjustment_id"], payload.get("tenant_id", "default"), payload["warehouse_id"], payload["sku"], payload["quantity_before"], payload["quantity_after"], payload["reason"], payload["approved_by"], payload["adjusted_at"], payload.get("manipulation_detected", False))


def update_dock_door_status(payload: dict):
	return _SERVICE.update_dock_door_status(payload["door_id"], payload.get("tenant_id", "default"), payload["door_number"], payload["warehouse_id"], payload["status"], payload.get("current_job_ref"))


def register_warehouse_agent(payload: dict):
	return _SERVICE.register_warehouse_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "warehouse operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
