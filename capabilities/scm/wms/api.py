"""Flask Blueprint REST API for Warehouse Management System (scm_wms)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import WarehouseManagementService

_log = logging.getLogger(__name__)

bp = Blueprint("scm_wms", __name__, url_prefix="/api/scm/wms")
_svc = WarehouseManagementService()


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@bp.get("/describe")
def describe():
	return jsonify(_run(_svc.describe()))


# ── Warehouses ────────────────────────────────────────────────────────────────

@bp.get("/warehouses")
def list_warehouses():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.list_warehouses(tenant_id=tenant)))


@bp.get("/warehouses/<warehouse_id>")
def get_warehouse(warehouse_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_warehouse(warehouse_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/warehouses")
def create_warehouse():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_warehouse(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Bins ──────────────────────────────────────────────────────────────────────

@bp.get("/bins")
def list_bins():
	tenant = request.args.get("tenant_id", "default")
	warehouse_id = request.args.get("warehouse_id")
	bin_type = request.args.get("bin_type")
	return jsonify(_run(_svc.list_bins(warehouse_id=warehouse_id, bin_type=bin_type, tenant_id=tenant)))


@bp.get("/bins/<bin_id>")
def get_bin(bin_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_bin(bin_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/bins")
def create_bin():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_bin(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/bins/<bin_id>")
def update_bin(bin_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_bin(bin_id, data, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/bins/<bin_id>")
def delete_bin(bin_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_bin(bin_id, tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Put-away tasks ────────────────────────────────────────────────────────────

@bp.get("/putaway-tasks")
def list_putaway():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_putaway_tasks(status=status, tenant_id=tenant)))


@bp.post("/putaway-tasks")
def create_putaway():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_putaway_task(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/putaway-tasks/<task_id>/complete")
def complete_putaway(task_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.complete_putaway_task(task_id, data["confirmed_bin_id"], data.get("completed_by", "system"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Pick tasks ────────────────────────────────────────────────────────────────

@bp.get("/pick-tasks")
def list_picks():
	tenant = request.args.get("tenant_id", "default")
	order_id = request.args.get("order_id")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_pick_tasks(order_id=order_id, status=status, tenant_id=tenant)))


@bp.post("/pick-tasks")
def create_pick():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_pick_task(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/pick-tasks/<task_id>/complete")
def complete_pick(task_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.complete_pick_task(task_id, data["picked_quantity"], data.get("completed_by", "system"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Pack tasks ────────────────────────────────────────────────────────────────

@bp.get("/pack-tasks")
def list_packs():
	tenant = request.args.get("tenant_id", "default")
	order_id = request.args.get("order_id")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_pack_tasks(order_id=order_id, status=status, tenant_id=tenant)))


@bp.post("/pack-tasks")
def create_pack():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_pack_task(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/pack-tasks/<task_id>/complete")
def complete_pack(task_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.complete_pack_task(task_id, data.get("cartons", []), data.get("total_weight_kg"), data.get("completed_by", "system"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Cycle counts ──────────────────────────────────────────────────────────────

@bp.get("/cycle-counts")
def list_cycle_counts():
	tenant = request.args.get("tenant_id", "default")
	warehouse_id = request.args.get("warehouse_id")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_cycle_counts(warehouse_id=warehouse_id, status=status, tenant_id=tenant)))


@bp.post("/cycle-counts")
def create_cycle_count():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_cycle_count(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/cycle-counts/<count_id>/submit")
def submit_cycle_count(count_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.submit_cycle_count_results(count_id, data.get("results", []), data.get("completed_by", "system"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Cross-docking ─────────────────────────────────────────────────────────────

@bp.get("/cross-docks")
def list_cross_docks():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_cross_docks(status=status, tenant_id=tenant)))


@bp.post("/cross-docks")
def create_cross_dock():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_cross_dock(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/cross-docks/<xd_id>/complete")
def complete_cross_dock(xd_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.complete_cross_dock(xd_id, data.get("completed_by", "system"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Slotting & inventory ──────────────────────────────────────────────────────

@bp.post("/slotting")
def run_slotting():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.run_slotting_optimisation(data["warehouse_id"], data.get("optimisation_objective", "pick_distance"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/inventory")
def get_inventory():
	tenant = request.args.get("tenant_id", "default")
	sku = request.args.get("sku")
	warehouse_id = request.args.get("warehouse_id")
	return jsonify(_run(_svc.get_inventory(sku=sku, warehouse_id=warehouse_id, tenant_id=tenant)))


# ── Analytics ─────────────────────────────────────────────────────────────────

@bp.get("/analytics")
def warehouse_analytics():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.warehouse_analytics(tenant_id=tenant)))


@bp.get("/audit-events")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant_id=tenant)))
