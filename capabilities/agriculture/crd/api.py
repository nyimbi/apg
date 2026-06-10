"""Agricultural Credit Scoring Flask Blueprint — agr_crd."""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request
from .service import AgriCreditService

_log = logging.getLogger(__name__)
bp = Blueprint("agr_crd", __name__, url_prefix="/api/agriculture/crd")
_svc: dict[str, AgriCreditService] = {}


def _get_svc(t: str = "default") -> AgriCreditService:
	if t not in _svc:
		_svc[t] = AgriCreditService(tenant_id=t)
	return _svc[t]


def _t() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@bp.get("/health")
async def health():
	return jsonify(await _get_svc(_t()).health_check()), 200


@bp.get("/profiles")
async def list_profiles():
	items = await _get_svc(_t()).list_profiles()
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/profiles")
async def create_profile():
	try:
		return jsonify(await _get_svc(_t()).create_profile(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/profiles/<profile_id>")
async def get_profile(profile_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_profile(profile_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/profiles/<profile_id>")
async def update_profile(profile_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_profile(profile_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/profiles/<profile_id>")
async def delete_profile(profile_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_profile(profile_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/score/<farmer_id>")
async def score_farmer(farmer_id: str):
	try:
		return jsonify(await _get_svc(_t()).score_farmer(farmer_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/loans")
async def list_loans():
	svc = _get_svc(_t())
	items = await svc.list_loans(farmer_id=request.args.get("farmer_id"), status=request.args.get("status"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/loans")
async def apply_loan():
	try:
		return jsonify(await _get_svc(_t()).apply_for_loan(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/loans/<loan_id>")
async def get_loan(loan_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_loan(loan_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/loans/<loan_id>")
async def update_loan(loan_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_loan(loan_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/loans/<loan_id>/repayment")
async def record_repayment(loan_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _get_svc(_t()).record_repayment(loan_id, float(body["amount"]))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/collateral")
async def list_collateral():
	items = await _get_svc(_t()).list_collateral(farmer_id=request.args.get("farmer_id"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/collateral")
async def create_collateral():
	try:
		return jsonify(await _get_svc(_t()).create_collateral(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/collateral/<col_id>")
async def delete_collateral(col_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_collateral(col_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/groups")
async def list_groups():
	items = await _get_svc(_t()).list_groups()
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/groups")
async def create_group():
	try:
		return jsonify(await _get_svc(_t()).create_group_loan(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/portfolio")
async def portfolio_summary():
	return jsonify(await _get_svc(_t()).get_portfolio_summary()), 200


@bp.get("/audit")
async def get_audit():
	events = await _get_svc(_t()).get_audit_events(int(request.args.get("limit", 100)))
	return jsonify({"events": events, "count": len(events)}), 200
