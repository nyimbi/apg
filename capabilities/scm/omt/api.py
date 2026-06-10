"""Flask Blueprint REST API for Order Management & Tracking (scm_omt)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import OrderManagementService

_log = logging.getLogger(__name__)

bp = Blueprint("scm_omt", __name__, url_prefix="/api/scm/omt")
_svc = OrderManagementService()


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


# ── Orders ────────────────────────────────────────────────────────────────────

@bp.get("/orders")
def list_orders():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	customer_id = request.args.get("customer_id")
	return jsonify(_run(_svc.list_orders(tenant_id=tenant, status=status, customer_id=customer_id)))


@bp.get("/orders/<order_id>")
def get_order(order_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_order(order_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/orders")
def create_order():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_order(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/orders/<order_id>")
def update_order(order_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_order(order_id, data, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/orders/<order_id>")
def delete_order(order_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_order(order_id, tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/orders/<order_id>/confirm")
def confirm_order(order_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	confirmed_by = data.get("confirmed_by", "system")
	try:
		return jsonify(_run(_svc.confirm_order(order_id, confirmed_by, tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/orders/<order_id>/cancel")
def cancel_order(order_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.cancel_order(order_id, data.get("reason", ""), data.get("cancelled_by", "system"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── ATP ───────────────────────────────────────────────────────────────────────

@bp.get("/atp")
def check_atp():
	tenant = request.args.get("tenant_id", "default")
	sku = request.args.get("sku", "")
	qty = float(request.args.get("quantity", 0))
	warehouse_id = request.args.get("warehouse_id")
	return jsonify(_run(_svc.check_atp(sku, qty, warehouse_id=warehouse_id, tenant_id=tenant)))


@bp.post("/atp")
def update_atp():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_atp(tenant_id=tenant, **data)))
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Backorders ────────────────────────────────────────────────────────────────

@bp.get("/backorders")
def list_backorders():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_backorders(tenant_id=tenant, status=status)))


@bp.post("/backorders")
def create_backorder():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_backorder(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/backorders/<bo_id>/fulfil")
def fulfil_backorder(bo_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	fulfilled_by = data.get("fulfilled_by", "system")
	try:
		return jsonify(_run(_svc.fulfil_backorder(bo_id, fulfilled_by, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Notifications ─────────────────────────────────────────────────────────────

@bp.post("/notifications")
def send_notification():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.send_notification(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/notifications")
def list_notifications():
	tenant = request.args.get("tenant_id", "default")
	order_id = request.args.get("order_id")
	return jsonify(_run(_svc.list_notifications(order_id=order_id, tenant_id=tenant)))


# ── Analytics ─────────────────────────────────────────────────────────────────

@bp.get("/analytics")
def order_analytics():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.order_analytics(tenant_id=tenant)))


@bp.get("/analytics/fulfilment-rate")
def fulfilment_rate():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.fulfilment_rate(tenant_id=tenant)))


@bp.get("/audit-events")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant_id=tenant)))
