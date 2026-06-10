"""Land Management Flask Blueprint — agr_lnd."""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request
from .service import LandManagementService

_log = logging.getLogger(__name__)
bp = Blueprint("agr_lnd", __name__, url_prefix="/api/agriculture/lnd")
_svc: dict[str, LandManagementService] = {}


def _get_svc(t: str = "default") -> LandManagementService:
	if t not in _svc:
		_svc[t] = LandManagementService(tenant_id=t)
	return _svc[t]


def _t() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@bp.get("/health")
async def health():
	return jsonify(await _get_svc(_t()).health_check()), 200


@bp.get("/parcels")
async def list_parcels():
	svc = _get_svc(_t())
	items = await svc.list_parcels(
		owner_id=request.args.get("owner_id"),
		county=request.args.get("county"),
		tenure_type=request.args.get("tenure_type"),
		status=request.args.get("status"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/parcels")
async def create_parcel():
	try:
		return jsonify(await _get_svc(_t()).create_parcel(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/parcels/<parcel_id>")
async def get_parcel(parcel_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_parcel(parcel_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/parcels/<parcel_id>")
async def update_parcel(parcel_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_parcel(parcel_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/parcels/<parcel_id>")
async def delete_parcel(parcel_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_parcel(parcel_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/owners/<owner_id>/holdings")
async def owner_holdings(owner_id: str):
	return jsonify(await _get_svc(_t()).get_owner_land_holdings(owner_id)), 200


@bp.get("/boundaries")
async def list_boundaries():
	items = await _get_svc(_t()).list_boundaries(parcel_id=request.args.get("parcel_id"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/boundaries")
async def capture_boundary():
	try:
		return jsonify(await _get_svc(_t()).capture_boundary(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/boundaries/<boundary_id>")
async def delete_boundary(boundary_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_boundary(boundary_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/titles")
async def list_titles():
	items = await _get_svc(_t()).list_titles(parcel_id=request.args.get("parcel_id"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/titles")
async def issue_title():
	try:
		return jsonify(await _get_svc(_t()).issue_title(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/titles/<title_id>/invalidate")
async def invalidate_title(title_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _get_svc(_t()).invalidate_title(title_id, body.get("reason", "unspecified"))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/titles/<title_id>")
async def delete_title(title_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_title(title_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/transfers")
async def list_transfers():
	svc = _get_svc(_t())
	items = await svc.list_transfers(parcel_id=request.args.get("parcel_id"), status=request.args.get("status"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/transfers")
async def initiate_transfer():
	try:
		return jsonify(await _get_svc(_t()).initiate_transfer(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/transfers/<transfer_id>")
async def update_transfer(transfer_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_transfer(transfer_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/transfers/<transfer_id>")
async def delete_transfer(transfer_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_transfer(transfer_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/registry-summary")
async def registry_summary():
	return jsonify(await _get_svc(_t()).get_land_registry_summary()), 200


@bp.get("/audit")
async def get_audit():
	events = await _get_svc(_t()).get_audit_events(int(request.args.get("limit", 100)))
	return jsonify({"events": events, "count": len(events)}), 200
