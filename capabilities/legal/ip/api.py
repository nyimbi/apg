"""Intellectual Property Registry — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import IntellectualPropertyService

_log = logging.getLogger(__name__)

bp = Blueprint("leg_ip", __name__, url_prefix="/api/legal/ip")
_svc: IntellectualPropertyService | None = None


def get_service() -> IntellectualPropertyService:
	global _svc
	if _svc is None:
		_svc = IntellectualPropertyService()
	return _svc


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
	return jsonify(_run(get_service().health_check()))


@bp.get("/describe")
def describe():
	return jsonify(_run(get_service().describe()))


@bp.get("/assets")
def list_assets():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_assets(
			tenant_id=tenant,
			asset_type=request.args.get("asset_type"),
			owner_id=request.args.get("owner_id"),
			jurisdiction=request.args.get("jurisdiction"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/assets/<asset_id>")
def get_asset(asset_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_asset(tenant, asset_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/assets")
def create_asset():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_asset(**body))), 201
	except Exception as exc:
		_log.error("create_asset: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.put("/assets/<asset_id>")
def update_asset(asset_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_asset(tenant, asset_id, **body)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/assets/<asset_id>")
def delete_asset(asset_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_asset(tenant, asset_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/assets/<asset_id>/register")
def register_asset(asset_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().register_asset(
			tenant, asset_id,
			body.get("registration_number", ""),
			body.get("registration_date", ""),
			body.get("expiry_date"),
		)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/renewals")
def list_renewals():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_renewals(tenant, request.args.get("asset_id")))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/renewals")
def create_renewal():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_renewal(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/renewals/<renewal_id>/confirm")
def confirm_renewal(renewal_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().confirm_renewal(
			tenant, renewal_id, body.get("new_expiry_date", ""), body.get("confirmed_by_id", ""),
		)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/licenses")
def list_licenses():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_licenses(tenant, request.args.get("asset_id")))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/licenses")
def create_license():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_license(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/licenses/<license_id>")
def terminate_license(license_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", request.args.get("tenant_id", "default"))
	try:
		return jsonify(_run(get_service().terminate_license(tenant, license_id, body.get("reason", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/royalties")
def list_royalties():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_royalties(tenant, request.args.get("license_id")))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/royalties")
def record_royalty():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().record_royalty(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/royalties/<royalty_id>/pay")
def pay_royalty(royalty_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().pay_royalty(tenant, royalty_id, body.get("payment_reference", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/expiring")
def expiring_assets():
	tenant = request.args.get("tenant_id", "default")
	days = int(request.args.get("days", 90))
	try:
		items = _run(get_service().expiring_assets(tenant, days))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/portfolio")
def portfolio_summary():
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().portfolio_summary(tenant)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/audit")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	limit = int(request.args.get("limit", 100))
	try:
		return jsonify(_run(get_service().get_audit_events(tenant, limit)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400
