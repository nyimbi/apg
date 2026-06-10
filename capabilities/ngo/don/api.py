"""Donor Relationship Management — Flask Blueprint with async REST endpoints."""
from __future__ import annotations

import logging
from decimal import Decimal

from flask import Blueprint, jsonify, request

from .service import DonorRelationshipService

_log = logging.getLogger(__name__)

bp = Blueprint("ngo_don", __name__, url_prefix="/api/ngo/don")

_svc: DonorRelationshipService | None = None


def _get_service() -> DonorRelationshipService:
	global _svc
	if _svc is None:
		_svc = DonorRelationshipService()
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
	return jsonify(_run(_get_service().health_check())), 200


@bp.get("/")
def list_donors():
	svc = _get_service()
	result = _run(svc.list_donors(status=request.args.get("status"), donor_type=request.args.get("donor_type")))
	return jsonify({"donors": result, "count": len(result)}), 200


@bp.get("/<donor_id>")
def get_donor(donor_id: str):
	try:
		return jsonify(_run(_get_service().get_donor(donor_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/")
def create_donor():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_donor(
			name=data["name"],
			donor_type=data.get("donor_type", "individual"),
			email=data.get("email", ""),
			phone=data.get("phone", ""),
			country=data.get("country", "KE"),
			address=data.get("address", ""),
			tax_id=data.get("tax_id", ""),
			notes=data.get("notes", ""),
			tags=data.get("tags", []),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/<donor_id>")
def update_donor(donor_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_get_service().update_donor(donor_id, **data))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/<donor_id>")
def delete_donor(donor_id: str):
	try:
		return jsonify(_run(_get_service().delete_donor(donor_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/search")
def search_donors():
	query = request.args.get("q", "")
	result = _run(_get_service().search_donors(query))
	return jsonify({"donors": result, "count": len(result)}), 200


@bp.get("/<donor_id>/communications")
def list_communications(donor_id: str):
	result = _run(_get_service().list_communications(donor_id=donor_id))
	return jsonify({"communications": result, "count": len(result)}), 200


@bp.post("/<donor_id>/communications")
def log_communication(donor_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().log_communication(
			donor_id=donor_id,
			subject=data["subject"],
			body=data.get("body", ""),
			staff_member=data["staff_member"],
			communication_date=data["communication_date"],
			channel=data.get("channel", "email"),
			direction=data.get("direction", "outbound"),
			tags=data.get("tags", []),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<donor_id>/pledges")
def list_pledges(donor_id: str):
	result = _run(_get_service().list_pledges(donor_id=donor_id))
	return jsonify({"pledges": result, "count": len(result)}), 200


@bp.post("/<donor_id>/pledges")
def create_pledge(donor_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_pledge(
			donor_id=donor_id,
			amount=Decimal(str(data["amount"])),
			pledge_date=data["pledge_date"],
			due_date=data["due_date"],
			currency=data.get("currency", "KES"),
			purpose=data.get("purpose", ""),
			frequency=data.get("frequency", "one_time"),
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<donor_id>/receipts")
def list_receipts(donor_id: str):
	result = _run(_get_service().list_receipts(donor_id=donor_id))
	return jsonify({"receipts": result, "count": len(result)}), 200


@bp.post("/<donor_id>/receipts")
def generate_receipt(donor_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().generate_receipt(
			donor_id=donor_id,
			amount=Decimal(str(data["amount"])),
			receipt_date=data["receipt_date"],
			reference=data["reference"],
			issued_by=data["issued_by"],
			pledge_id=data.get("pledge_id"),
			currency=data.get("currency", "KES"),
			payment_method=data.get("payment_method", "bank_transfer"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<donor_id>/stewardship")
def list_stewardship_plans(donor_id: str):
	# filter client-side from full list
	result = _run(_get_service().list_stewardship_plans())
	result = [p for p in result if p["donor_id"] == donor_id]
	return jsonify({"plans": result, "count": len(result)}), 200


@bp.post("/<donor_id>/stewardship")
def create_stewardship_plan(donor_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_stewardship_plan(
			donor_id=donor_id,
			tier=data.get("tier", "standard"),
			touchpoints_per_year=int(data.get("touchpoints_per_year", 4)),
			assigned_to=data.get("assigned_to", ""),
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<donor_id>/history")
def donor_giving_history(donor_id: str):
	try:
		return jsonify(_run(_get_service().donor_giving_history(donor_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/portfolio/summary")
def portfolio_summary():
	return jsonify(_run(_get_service().portfolio_summary())), 200


@bp.get("/portfolio/retention")
def retention_analysis():
	return jsonify(_run(_get_service().retention_analysis())), 200


@bp.get("/pledges/overdue")
def overdue_pledges():
	result = _run(_get_service().overdue_pledges())
	return jsonify({"pledges": result, "count": len(result)}), 200


@bp.get("/audit-events")
def get_audit_events():
	limit = int(request.args.get("limit", 100))
	result = _run(_get_service().get_audit_events(limit=limit))
	return jsonify({"events": result, "count": len(result)}), 200
