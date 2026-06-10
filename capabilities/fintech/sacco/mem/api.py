"""Flask Blueprint REST API for SACCO Member Registry."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import SaccoMemberRegistryService

_log = logging.getLogger(__name__)

bp = Blueprint("sacco_mem", __name__, url_prefix="/api/fintech/sacco/mem")
_svc = SaccoMemberRegistryService()


def _run(coro: Any) -> Any:
	"""Run an async coroutine from a sync Flask view."""
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


# ── Health ────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
	result = _run(_svc.health_check())
	return jsonify(result), 200


# ── Members ───────────────────────────────────────────────────────────────────

@bp.get("/members")
def list_members():
	params = request.args
	result = _run(_svc.list_members(
		tenant_id=_tenant(),
		status=params.get("status"),
		membership_type=params.get("membership_type"),
		kyc_status=params.get("kyc_status"),
		county=params.get("county"),
	))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/members/<member_id>")
def get_member(member_id: str):
	try:
		result = _run(_svc.get_member(member_id, tenant_id=_tenant()))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("get_member error: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.post("/members")
def create_member():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.create_member(
			full_name=body["full_name"],
			national_id=body["national_id"],
			phone=body["phone"],
			date_of_birth=body["date_of_birth"],
			gender=body.get("gender", "M"),
			county=body["county"],
			tenant_id=_tenant(),
			email=body.get("email"),
			membership_type=body.get("membership_type", "ordinary"),
			sub_county=body.get("sub_county"),
			postal_address=body.get("postal_address"),
			occupation=body.get("occupation"),
			employer=body.get("employer"),
			monthly_income=body.get("monthly_income"),
			entry_fee=body.get("entry_fee", 0.0),
			minimum_shares=body.get("minimum_shares", 1),
			next_of_kin_name=body.get("next_of_kin_name"),
			next_of_kin_phone=body.get("next_of_kin_phone"),
			next_of_kin_relationship=body.get("next_of_kin_relationship"),
			referred_by=body.get("referred_by"),
		))
		return jsonify(result), 201
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 422
	except Exception as exc:
		_log.error("create_member error: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.put("/members/<member_id>")
def update_member(member_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.update_member(member_id, tenant_id=_tenant(), **body))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("update_member error: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.delete("/members/<member_id>")
def delete_member(member_id: str):
	try:
		result = _run(_svc.delete_member(member_id, tenant_id=_tenant()))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("delete_member error: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.get("/members/search")
def search_members():
	query = request.args.get("q", "")
	result = _run(_svc.search_members(query, tenant_id=_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200


# ── KYC ───────────────────────────────────────────────────────────────────────

@bp.get("/kyc")
def list_kyc():
	result = _run(_svc.list_kyc_records(tenant_id=_tenant(), status=request.args.get("status")))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/kyc")
def submit_kyc():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.submit_kyc(
			member_id=body["member_id"],
			document_type=body["document_type"],
			document_number=body["document_number"],
			document_front_ref=body["document_front_ref"],
			submitted_by=body["submitted_by"],
			tenant_id=_tenant(),
			document_back_ref=body.get("document_back_ref"),
			selfie_ref=body.get("selfie_ref"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/kyc/<kyc_id>/approve")
def approve_kyc(kyc_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.approve_kyc(kyc_id, verified_by=body["verified_by"], tenant_id=_tenant(), notes=body.get("notes")))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/kyc/<kyc_id>/reject")
def reject_kyc(kyc_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.reject_kyc(kyc_id, verified_by=body["verified_by"], rejection_reason=body["rejection_reason"], tenant_id=_tenant()))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Share Capital ─────────────────────────────────────────────────────────────

@bp.post("/shares/purchase")
def purchase_shares():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.purchase_shares(
			member_id=body["member_id"],
			shares=body["shares"],
			share_value=body["share_value"],
			payment_reference=body["payment_reference"],
			recorded_by=body["recorded_by"],
			tenant_id=_tenant(),
			payment_method=body.get("payment_method", "cash"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/shares/transfer")
def transfer_shares():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.transfer_shares(
			from_member_id=body["from_member_id"],
			to_member_id=body["to_member_id"],
			shares=body["shares"],
			transfer_reason=body["transfer_reason"],
			approved_by=body["approved_by"],
			tenant_id=_tenant(),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Exits ─────────────────────────────────────────────────────────────────────

@bp.post("/exits")
def initiate_exit():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.initiate_exit(
			member_id=body["member_id"],
			exit_reason=body["exit_reason"],
			exit_date=body["exit_date"],
			processed_by=body["processed_by"],
			tenant_id=_tenant(),
			notes=body.get("notes"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/exits/<exit_id>/complete")
def complete_exit(exit_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.complete_exit(exit_id, approved_by=body["approved_by"], settlement_reference=body["settlement_reference"], tenant_id=_tenant()))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Summary & Audit ───────────────────────────────────────────────────────────

@bp.get("/summary")
def summary():
	result = _run(_svc.membership_summary(tenant_id=_tenant()))
	return jsonify(result), 200


@bp.get("/audit")
def audit_events():
	result = _run(_svc.get_audit_events(tenant_id=_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200
