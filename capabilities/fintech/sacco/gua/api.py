"""Flask Blueprint REST API for SACCO Guarantor Management."""
from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any

from flask import Blueprint, jsonify, request

from .service import GuarantorService

_log = logging.getLogger(__name__)

bp = Blueprint("sacco_gua", __name__, url_prefix="/api/fintech/sacco/gua")
_svc = GuarantorService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _dec(v: Any) -> Decimal:
	return Decimal(str(v))


# ── Health ────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check())), 200


# ── Eligibility & Exposure ────────────────────────────────────────────────────

@bp.post("/eligibility")
def check_eligibility():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.check_guarantor_eligibility(
			tenant_id=_tenant(),
			member_id=body["member_id"],
			amount_to_guarantee=_dec(body["amount_to_guarantee"]),
		))
		return jsonify(result), 200
	except (KeyError, ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/exposure/<member_id>")
def get_exposure(member_id: str):
	return jsonify(_run(_svc.get_guarantor_exposure(_tenant(), member_id))), 200


@bp.post("/exposure-limit")
def set_exposure_limit():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.set_exposure_limit(
			tenant_id=_tenant(),
			member_id=body["member_id"],
			limit=_dec(body["limit"]),
			set_by=body["set_by"],
		))
		return jsonify(result), 200
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Requests ──────────────────────────────────────────────────────────────────

@bp.post("/requests")
def request_guarantee():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.request_guarantee(
			tenant_id=_tenant(),
			loan_id=body["loan_id"],
			guarantor_member_id=body["guarantor_member_id"],
			amount_to_guarantee=_dec(body["amount_to_guarantee"]),
			loan_applicant_message=body.get("loan_applicant_message"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422
	except Exception as exc:
		_log.exception("request_guarantee failed")
		return jsonify({"error": str(exc)}), 500


@bp.get("/requests")
def list_requests():
	result = _run(_svc.list_guarantee_requests(
		tenant_id=_tenant(),
		loan_id=request.args.get("loan_id"),
		guarantor_id=request.args.get("guarantor_id"),
		status=request.args.get("status"),
	))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/requests/<request_id>")
def get_request(request_id: str):
	try:
		return jsonify(_run(_svc.get_guarantee_request(_tenant(), request_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/requests/<request_id>/accept")
def accept_guarantee(request_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.accept_guarantee(
			tenant_id=_tenant(),
			guarantee_request_id=request_id,
			guarantor_member_id=body["guarantor_member_id"],
			pin_verified=bool(body.get("pin_verified", False)),
			acceptance_notes=body.get("acceptance_notes"),
		))
		return jsonify(result), 200
	except (KeyError, ValueError, PermissionError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/requests/<request_id>/decline")
def decline_guarantee(request_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.decline_guarantee(
			tenant_id=_tenant(),
			guarantee_request_id=request_id,
			guarantor_member_id=body["guarantor_member_id"],
			decline_reason=body["decline_reason"],
		))
		return jsonify(result), 200
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/requests/<request_id>/cancel")
def cancel_request(request_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.cancel_guarantee_request(
			tenant_id=_tenant(),
			guarantee_request_id=request_id,
			cancelled_by=body["cancelled_by"],
			reason=body["reason"],
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Active Guarantees ─────────────────────────────────────────────────────────

@bp.get("/guarantees")
def list_guarantees():
	loan_id = request.args.get("loan_id")
	guarantor_id = request.args.get("guarantor_id")
	t = _tenant()
	if loan_id:
		result = _run(_svc.get_loan_guarantors(t, loan_id))
	elif guarantor_id:
		result = _run(_svc.get_guarantor_history(t, guarantor_id)).get("guarantees", [])
	else:
		# return all active for tenant
		result = [
			g for g in _svc._guarantees.values()
			if g["tenant_id"] == t
		]
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/guarantees/<guarantee_id>/release")
def release_guarantee(guarantee_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.release_guarantee(
			tenant_id=_tenant(),
			guarantee_id=guarantee_id,
			release_reason=body["release_reason"],
			released_by=body.get("released_by", "api"),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/guarantees/<guarantee_id>/substitute")
def substitute_guarantor(guarantee_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.substitute_guarantor(
			tenant_id=_tenant(),
			guarantee_id=guarantee_id,
			new_guarantor_id=body["new_guarantor_id"],
			reason=body["reason"],
			approved_by=body["approved_by"],
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/guarantees/<guarantee_id>/call")
def call_guarantee(guarantee_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.call_guarantee(
			tenant_id=_tenant(),
			guarantee_id=guarantee_id,
			amount_called=_dec(body["amount_called"]),
			reason=body["reason"],
		))
		return jsonify(result), 200
	except (KeyError, ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/guarantees/<guarantee_id>/notice")
def send_notice(guarantee_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.send_guarantee_notice(
			tenant_id=_tenant(),
			guarantee_id=guarantee_id,
			notice_type=body["notice_type"],
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Member views ──────────────────────────────────────────────────────────────

@bp.get("/members/<member_id>/history")
def member_history(member_id: str):
	return jsonify(_run(_svc.get_guarantor_history(_tenant(), member_id))), 200


@bp.get("/members/<member_id>/called")
def called_guarantees(member_id: str):
	result = _run(_svc.get_called_guarantees(_tenant(), member_id))
	return jsonify({"items": result, "total": len(result)}), 200


# ── Portfolio & Operations ────────────────────────────────────────────────────

@bp.get("/at-risk")
def at_risk():
	result = _run(_svc.get_at_risk_guarantees(_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/metrics")
def metrics():
	return jsonify(_run(_svc.get_guarantee_portfolio_metrics(_tenant()))), 200


@bp.post("/process-releases")
def process_releases():
	return jsonify(_run(_svc.process_automatic_releases(_tenant()))), 200


@bp.get("/gl-entries")
def gl_entries():
	result = _run(_svc.get_gl_entries(
		tenant_id=_tenant(),
		guarantee_id=request.args.get("guarantee_id"),
	))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/audit")
def audit():
	result = _run(_svc.get_audit_events(_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200
