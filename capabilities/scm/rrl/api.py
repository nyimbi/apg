"""Flask Blueprint REST API for Returns & Reverse Logistics (scm_rrl)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import ReturnsService

_log = logging.getLogger(__name__)

bp = Blueprint("scm_rrl", __name__, url_prefix="/api/scm/rrl")
_svc = ReturnsService()


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


# ── RMAs ──────────────────────────────────────────────────────────────────────

@bp.get("/rmas")
def list_rmas():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	customer_id = request.args.get("customer_id")
	return jsonify(_run(_svc.list_rmas(tenant_id=tenant, status=status, customer_id=customer_id)))


@bp.get("/rmas/<rma_id>")
def get_rma(rma_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_rma(rma_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/rmas")
def create_rma():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_rma(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/rmas/<rma_id>")
def update_rma(rma_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_rma(rma_id, data, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/rmas/<rma_id>")
def delete_rma(rma_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_rma(rma_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/rmas/<rma_id>/approve")
def approve_rma(rma_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.approve_rma(rma_id, data.get("approved_by", "system"), data.get("notes"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/rmas/<rma_id>/reject")
def reject_rma(rma_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.reject_rma(rma_id, data.get("rejected_by", "system"), data.get("reason", ""), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/rmas/<rma_id>/receive")
def receive_return(rma_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.receive_return(rma_id, data.get("received_by", "system"), data.get("condition_notes"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/rmas/<rma_id>/resolve")
def resolve_rma(rma_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.resolve_rma(rma_id, data["resolution"], data.get("resolved_by", "system"), data.get("notes"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Refurbishments ────────────────────────────────────────────────────────────

@bp.get("/refurbishments")
def list_refurbishments():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_refurbishments(tenant_id=tenant, status=status)))


@bp.post("/refurbishments")
def create_refurbishment():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_refurbishment(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/refurbishments/<refurb_id>/complete")
def complete_refurbishment(refurb_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.complete_refurbishment(refurb_id, data["condition_after"], data["actual_cost"], data.get("completed_by", "system"), tenant_id=tenant)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Disposals ─────────────────────────────────────────────────────────────────

@bp.get("/disposals")
def list_disposals():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_disposals(tenant_id=tenant, status=status)))


@bp.post("/disposals")
def create_disposal():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_disposal(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Credit notes ──────────────────────────────────────────────────────────────

@bp.get("/credit-notes")
def list_credit_notes():
	tenant = request.args.get("tenant_id", "default")
	customer_id = request.args.get("customer_id")
	return jsonify(_run(_svc.list_credit_notes(customer_id=customer_id, tenant_id=tenant)))


@bp.post("/credit-notes")
def issue_credit_note():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.issue_credit_note(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Analytics ─────────────────────────────────────────────────────────────────

@bp.get("/analytics")
def returns_analytics():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.returns_analytics(tenant_id=tenant)))


@bp.get("/audit-events")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant_id=tenant)))
