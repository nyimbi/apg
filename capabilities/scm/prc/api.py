"""Flask Blueprint REST API for Procurement Management (scm_prc)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import ProcurementService

_log = logging.getLogger(__name__)

bp = Blueprint("scm_prc", __name__, url_prefix="/api/scm/prc")
_svc = ProcurementService()


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


# ── RFQs ──────────────────────────────────────────────────────────────────────

@bp.get("/rfqs")
def list_rfqs():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_rfqs(tenant_id=tenant, status=status)))


@bp.get("/rfqs/<rfq_id>")
def get_rfq(rfq_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_rfq(rfq_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/rfqs")
def create_rfq():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_rfq(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/rfqs/<rfq_id>/issue")
def issue_rfq(rfq_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	issued_by = data.get("issued_by", "system")
	try:
		return jsonify(_run(_svc.issue_rfq(rfq_id, issued_by, tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/rfqs/<rfq_id>/award")
def award_rfq(rfq_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.award_rfq(rfq_id, data["winning_vendor_id"], data.get("awarded_by", "system"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Purchase Orders ───────────────────────────────────────────────────────────

@bp.get("/purchase-orders")
def list_pos():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	vendor_id = request.args.get("vendor_id")
	return jsonify(_run(_svc.list_purchase_orders(tenant_id=tenant, status=status, vendor_id=vendor_id)))


@bp.get("/purchase-orders/<po_id>")
def get_po(po_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_purchase_order(po_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/purchase-orders")
def create_po():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_purchase_order(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/purchase-orders/<po_id>")
def update_po(po_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_purchase_order(po_id, data, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/purchase-orders/<po_id>")
def delete_po(po_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_purchase_order(po_id, tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/purchase-orders/<po_id>/send")
def send_po(po_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	sent_by = data.get("sent_by", "system")
	try:
		return jsonify(_run(_svc.send_purchase_order(po_id, sent_by, tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/purchase-orders/<po_id>/receive")
def receive_po(po_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.receive_purchase_order(po_id, tenant_id=tenant, **data)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Three-way match ───────────────────────────────────────────────────────────

@bp.get("/three-way-matches")
def list_matches():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_three_way_matches(tenant_id=tenant, status=status)))


@bp.post("/three-way-matches")
def create_match():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_three_way_match(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Vendor evaluations ────────────────────────────────────────────────────────

@bp.get("/vendor-evaluations")
def list_evaluations():
	tenant = request.args.get("tenant_id", "default")
	vendor_id = request.args.get("vendor_id")
	return jsonify(_run(_svc.list_vendor_evaluations(vendor_id=vendor_id, tenant_id=tenant)))


@bp.post("/vendor-evaluations")
def create_evaluation():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_vendor_evaluation(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Contracts ─────────────────────────────────────────────────────────────────

@bp.get("/contracts")
def list_contracts():
	tenant = request.args.get("tenant_id", "default")
	vendor_id = request.args.get("vendor_id")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_contracts(vendor_id=vendor_id, status=status, tenant_id=tenant)))


@bp.post("/contracts")
def create_contract():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_contract(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Analytics ─────────────────────────────────────────────────────────────────

@bp.get("/analytics/spend")
def spend_analytics():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.spend_analytics(tenant_id=tenant)))


@bp.get("/analytics/dashboard")
def procurement_dashboard():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.procurement_dashboard(tenant_id=tenant)))


@bp.get("/audit-events")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant_id=tenant)))
