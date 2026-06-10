"""Grant Management — Flask Blueprint with async REST endpoints."""
from __future__ import annotations

import logging
from decimal import Decimal

from flask import Blueprint, jsonify, request

from .service import GrantManagementService

_log = logging.getLogger(__name__)

bp = Blueprint("ngo_grn", __name__, url_prefix="/api/ngo/grn")

_svc: GrantManagementService | None = None


def _get_service() -> GrantManagementService:
	global _svc
	if _svc is None:
		_svc = GrantManagementService()
	return _svc


def _run(coro):
	import asyncio
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as pool:
				future = pool.submit(asyncio.run, coro)
				return future.result()
		return loop.run_until_complete(coro)
	except Exception as exc:
		_log.error("async execution error: %s", exc)
		raise


@bp.get("/health")
def health():
	result = _run(_get_service().health_check())
	return jsonify(result), 200


@bp.get("/")
def list_grants():
	svc = _get_service()
	status = request.args.get("status")
	sector = request.args.get("sector")
	result = _run(svc.list_grants(status=status, sector=sector))
	return jsonify({"grants": result, "count": len(result)}), 200


@bp.get("/<grant_id>")
def get_grant(grant_id: str):
	try:
		result = _run(_get_service().get_grant(grant_id))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/")
def create_grant():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_grant(
			title=data["title"],
			donor_reference=data["donor_reference"],
			amount=Decimal(str(data["amount"])),
			start_date=data["start_date"],
			end_date=data["end_date"],
			currency=data.get("currency", "KES"),
			sector=data.get("sector", ""),
			country=data.get("country", "KE"),
			programme_id=data.get("programme_id"),
			contact_person=data.get("contact_person", ""),
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/<grant_id>")
def update_grant(grant_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().update_grant(grant_id, **data))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/<grant_id>")
def delete_grant(grant_id: str):
	try:
		result = _run(_get_service().delete_grant(grant_id))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/<grant_id>/activate")
def activate_grant(grant_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().activate_grant(grant_id, approved_by=data.get("approved_by", "")))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/<grant_id>/close")
def close_grant(grant_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().close_grant(grant_id, closed_by=data.get("closed_by", ""), reason=data.get("reason", "")))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<grant_id>/proposals")
def list_proposals(grant_id: str):
	result = _run(_get_service().list_proposals(grant_id=grant_id))
	return jsonify({"proposals": result, "count": len(result)}), 200


@bp.post("/<grant_id>/proposals")
def create_proposal(grant_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_proposal(
			grant_id=grant_id,
			title=data["title"],
			narrative=data.get("narrative", ""),
			budget=Decimal(str(data["budget"])),
			submitted_by=data["submitted_by"],
			deadline=data["deadline"],
			currency=data.get("currency", "KES"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<grant_id>/budget-lines")
def list_budget_lines(grant_id: str):
	result = _run(_get_service().list_budget_lines(grant_id))
	return jsonify({"budget_lines": result, "count": len(result)}), 200


@bp.post("/<grant_id>/budget-lines")
def create_budget_line(grant_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_budget_line(
			grant_id=grant_id,
			category=data["category"],
			description=data.get("description", ""),
			amount=Decimal(str(data["amount"])),
			currency=data.get("currency", "KES"),
			period=data.get("period", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<grant_id>/disbursements")
def list_disbursements(grant_id: str):
	result = _run(_get_service().list_disbursements(grant_id=grant_id))
	return jsonify({"disbursements": result, "count": len(result)}), 200


@bp.post("/<grant_id>/disbursements")
def create_disbursement(grant_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_disbursement(
			grant_id=grant_id,
			amount=Decimal(str(data["amount"])),
			disbursement_date=data["disbursement_date"],
			reference=data["reference"],
			approved_by=data["approved_by"],
			currency=data.get("currency", "KES"),
			payment_method=data.get("payment_method", "bank_transfer"),
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<grant_id>/compliance-reports")
def list_compliance_reports(grant_id: str):
	result = _run(_get_service().list_compliance_reports(grant_id=grant_id))
	return jsonify({"reports": result, "count": len(result)}), 200


@bp.post("/<grant_id>/compliance-reports")
def create_compliance_report(grant_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_compliance_report(
			grant_id=grant_id,
			report_type=data["report_type"],
			period_start=data["period_start"],
			period_end=data["period_end"],
			submitted_by=data["submitted_by"],
			narrative=data.get("narrative", ""),
			attachments=data.get("attachments", []),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<grant_id>/audit-findings")
def list_audit_findings(grant_id: str):
	result = _run(_get_service().list_audit_findings(grant_id=grant_id))
	return jsonify({"findings": result, "count": len(result)}), 200


@bp.post("/<grant_id>/audit-findings")
def create_audit_finding(grant_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_audit_finding(
			grant_id=grant_id,
			finding_type=data["finding_type"],
			description=data["description"],
			auditor=data["auditor"],
			audit_date=data["audit_date"],
			severity=data.get("severity", "medium"),
			recommendations=data.get("recommendations", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<grant_id>/summary")
def grant_summary(grant_id: str):
	try:
		result = _run(_get_service().generate_donor_report(grant_id))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/portfolio/summary")
def portfolio_summary():
	result = _run(_get_service().grant_portfolio_summary())
	return jsonify(result), 200


@bp.get("/audit-events")
def get_audit_events():
	limit = int(request.args.get("limit", 100))
	result = _run(_get_service().get_audit_events(limit=limit))
	return jsonify({"events": result, "count": len(result)}), 200
