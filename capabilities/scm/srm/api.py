"""Flask Blueprint REST API for Supplier Relationship Management (scm_srm)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import SupplierRelationshipService

_log = logging.getLogger(__name__)

bp = Blueprint("scm_srm", __name__, url_prefix="/api/scm/srm")
_svc = SupplierRelationshipService()


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


# ── Suppliers ─────────────────────────────────────────────────────────────────

@bp.get("/suppliers")
def list_suppliers():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	category = request.args.get("category")
	preferred_only = request.args.get("preferred_only", "false").lower() == "true"
	return jsonify(_run(_svc.list_suppliers(status=status, category=category, preferred_only=preferred_only, tenant_id=tenant)))


@bp.get("/suppliers/<supplier_id>")
def get_supplier(supplier_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_supplier(supplier_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/suppliers")
def create_supplier():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_supplier(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/suppliers/<supplier_id>")
def update_supplier(supplier_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_supplier(supplier_id, data, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/suppliers/<supplier_id>")
def delete_supplier(supplier_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_supplier(supplier_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/suppliers/<supplier_id>/approve")
def approve_supplier(supplier_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	approved_by = data.get("approved_by", "system")
	try:
		return jsonify(_run(_svc.approve_supplier(supplier_id, approved_by, tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/suppliers/<supplier_id>/suspend")
def suspend_supplier(supplier_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.suspend_supplier(supplier_id, data.get("reason", ""), data.get("suspended_by", "system"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/suppliers/<supplier_id>/preferred")
def set_preferred(supplier_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.set_preferred_status(supplier_id, data.get("preferred", True), data.get("reason", ""), data.get("set_by", "system"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Scorecards ────────────────────────────────────────────────────────────────

@bp.get("/scorecards")
def list_scorecards():
	tenant = request.args.get("tenant_id", "default")
	supplier_id = request.args.get("supplier_id")
	period = request.args.get("period")
	return jsonify(_run(_svc.list_scorecards(supplier_id=supplier_id, period=period, tenant_id=tenant)))


@bp.post("/scorecards")
def create_scorecard():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_scorecard(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Risk assessments ──────────────────────────────────────────────────────────

@bp.get("/risk-assessments")
def list_risks():
	tenant = request.args.get("tenant_id", "default")
	supplier_id = request.args.get("supplier_id")
	risk_level = request.args.get("risk_level")
	return jsonify(_run(_svc.list_risk_assessments(supplier_id=supplier_id, risk_level=risk_level, tenant_id=tenant)))


@bp.post("/risk-assessments")
def create_risk():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_risk_assessment(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/risk-assessments/<assessment_id>/review")
def review_risk(assessment_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.review_risk_assessment(assessment_id, data.get("reviewed_by", "system"), data.get("outcome", ""), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Collaboration ─────────────────────────────────────────────────────────────

@bp.get("/messages")
def list_messages():
	tenant = request.args.get("tenant_id", "default")
	supplier_id = request.args.get("supplier_id")
	return jsonify(_run(_svc.list_collaboration_messages(supplier_id=supplier_id, tenant_id=tenant)))


@bp.post("/messages")
def send_message():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.send_collaboration_message(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Performance reviews ───────────────────────────────────────────────────────

@bp.get("/performance-reviews")
def list_reviews():
	tenant = request.args.get("tenant_id", "default")
	supplier_id = request.args.get("supplier_id")
	return jsonify(_run(_svc.list_performance_reviews(supplier_id=supplier_id, tenant_id=tenant)))


@bp.post("/performance-reviews")
def create_review():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_performance_review(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Analytics ─────────────────────────────────────────────────────────────────

@bp.get("/analytics")
def supplier_analytics():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.supplier_analytics(tenant_id=tenant)))


@bp.get("/audit-events")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant_id=tenant)))
