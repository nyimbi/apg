"""Cooperative Management Flask Blueprint — agr_coo."""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request
from .service import CooperativeManagementService

_log = logging.getLogger(__name__)
bp = Blueprint("agr_coo", __name__, url_prefix="/api/agriculture/coo")
_svc: dict[str, CooperativeManagementService] = {}


def _get_svc(t: str = "default") -> CooperativeManagementService:
	if t not in _svc:
		_svc[t] = CooperativeManagementService(tenant_id=t)
	return _svc[t]


def _t() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@bp.get("/health")
async def health():
	return jsonify(await _get_svc(_t()).health_check()), 200


@bp.get("/coops")
async def list_coops():
	items = await _get_svc(_t()).list_coops(region=request.args.get("region"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/coops")
async def create_coop():
	try:
		return jsonify(await _get_svc(_t()).create_coop(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/coops/<coop_id>")
async def get_coop(coop_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_coop(coop_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/coops/<coop_id>")
async def update_coop(coop_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_coop(coop_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/coops/<coop_id>/summary")
async def coop_summary(coop_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_coop_summary(coop_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/members")
async def list_members():
	svc = _get_svc(_t())
	items = await svc.list_members(coop_id=request.args.get("coop_id"), status=request.args.get("status"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/members")
async def create_member():
	try:
		return jsonify(await _get_svc(_t()).create_member(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/members/<member_id>")
async def get_member(member_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_member(member_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/members/<member_id>")
async def update_member(member_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_member(member_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/members/<member_id>/statement")
async def member_statement(member_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_member_statement(member_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/members/transfer-shares")
async def transfer_shares():
	try:
		body = request.get_json(force=True) or {}
		result = await _get_svc(_t()).transfer_shares(body["from_member_id"], body["to_member_id"], int(body["shares"]))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/input-pools")
async def list_pools():
	svc = _get_svc(_t())
	items = await svc.list_input_pools(coop_id=request.args.get("coop_id"), season=request.args.get("season"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/input-pools")
async def create_pool():
	try:
		return jsonify(await _get_svc(_t()).create_input_pool(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/input-pools/<pool_id>/allocate")
async def allocate_pool(pool_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _get_svc(_t()).allocate_from_pool(pool_id, body["member_id"], float(body["quantity"]))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/dividends")
async def list_dividends():
	items = await _get_svc(_t()).list_dividends(coop_id=request.args.get("coop_id"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/dividends")
async def allocate_dividends():
	try:
		return jsonify(await _get_svc(_t()).allocate_dividends(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/annual-returns")
async def list_returns():
	items = await _get_svc(_t()).list_annual_returns(coop_id=request.args.get("coop_id"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/annual-returns")
async def file_return():
	try:
		return jsonify(await _get_svc(_t()).file_annual_return(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/audit")
async def get_audit():
	events = await _get_svc(_t()).get_audit_events(int(request.args.get("limit", 100)))
	return jsonify({"events": events, "count": len(events)}), 200
