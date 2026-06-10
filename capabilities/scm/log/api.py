"""Flask Blueprint REST API for Logistics & Transportation (scm_log)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import LogisticsService

_log = logging.getLogger(__name__)

bp = Blueprint("scm_log", __name__, url_prefix="/api/scm/log")
_svc = LogisticsService()


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


# ── Carriers ──────────────────────────────────────────────────────────────────

@bp.get("/carriers")
def list_carriers():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_carriers(tenant_id=tenant, status=status)))


@bp.get("/carriers/<carrier_id>")
def get_carrier(carrier_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_carrier(carrier_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/carriers")
def create_carrier():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_carrier(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/carriers/<carrier_id>")
def update_carrier(carrier_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_carrier(carrier_id, data, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/carriers/<carrier_id>")
def delete_carrier(carrier_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_carrier(carrier_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Shipments ─────────────────────────────────────────────────────────────────

@bp.get("/shipments")
def list_shipments():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	carrier_id = request.args.get("carrier_id")
	return jsonify(_run(_svc.list_shipments(tenant_id=tenant, status=status, carrier_id=carrier_id)))


@bp.get("/shipments/<shipment_id>")
def get_shipment(shipment_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_shipment(shipment_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/shipments")
def create_shipment():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_shipment(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/shipments/<shipment_id>")
def update_shipment(shipment_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_shipment(shipment_id, data, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/shipments/<shipment_id>/book")
def book_shipment(shipment_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.book_shipment(shipment_id, tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/shipments/<shipment_id>/cancel")
def cancel_shipment(shipment_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	reason = data.get("reason", "")
	try:
		return jsonify(_run(_svc.cancel_shipment(shipment_id, reason, tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/shipments/<shipment_id>/tracking")
def get_tracking(shipment_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_shipment_tracking(shipment_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/shipments/<shipment_id>/tracking")
def add_tracking(shipment_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.add_tracking_event(shipment_id, tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Freight audits ────────────────────────────────────────────────────────────

@bp.get("/freight-audits")
def list_freight_audits():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_freight_audits(tenant_id=tenant, status=status)))


@bp.post("/freight-audits")
def create_freight_audit():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_freight_audit(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/freight-audits/<audit_id>/resolve")
def resolve_freight_audit(audit_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.resolve_freight_audit(audit_id, tenant_id=tenant, **data)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Routes ────────────────────────────────────────────────────────────────────

@bp.get("/routes")
def list_routes():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.list_routes(tenant_id=tenant)))


@bp.post("/routes")
def create_route():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_route(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/routes/<route_id>/optimise")
def optimise_route(route_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.optimise_route(route_id, data, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Analytics ─────────────────────────────────────────────────────────────────

@bp.get("/analytics/shipments")
def shipment_analytics():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.shipment_analytics(tenant_id=tenant)))


@bp.get("/analytics/freight-costs")
def freight_cost_summary():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.freight_cost_summary(tenant_id=tenant)))


@bp.get("/audit-events")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant_id=tenant)))
