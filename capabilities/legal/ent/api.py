"""Entity & Corporate Secretary — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import EntityCorporateSecretaryService

_log = logging.getLogger(__name__)

bp = Blueprint("leg_ent", __name__, url_prefix="/api/legal/ent")
_svc: EntityCorporateSecretaryService | None = None


def get_service() -> EntityCorporateSecretaryService:
	global _svc
	if _svc is None:
		_svc = EntityCorporateSecretaryService()
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


@bp.get("/entities")
def list_entities():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_entities(
			tenant_id=tenant,
			entity_type=request.args.get("entity_type"),
			jurisdiction=request.args.get("jurisdiction"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/entities/<entity_id>")
def get_entity(entity_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_entity(tenant, entity_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/entities")
def create_entity():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_entity(**body))), 201
	except Exception as exc:
		_log.error("create_entity: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.put("/entities/<entity_id>")
def update_entity(entity_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_entity(tenant, entity_id, **body)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/entities/<entity_id>")
def delete_entity(entity_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_entity(tenant, entity_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/entities/<entity_id>/directors")
def list_directors(entity_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_directors(tenant, entity_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/directors")
def appoint_director():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().appoint_director(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/directors/<director_id>")
def update_director(director_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_director(tenant, director_id, **body)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/directors/<director_id>")
def remove_director(director_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().remove_director(tenant, director_id, body.get("cessation_date", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/entities/<entity_id>/shareholders")
def list_shareholders(entity_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_shareholders(tenant, entity_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/shareholders")
def register_shareholder():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().register_shareholder(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/shareholders/transfer")
def transfer_shares():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().transfer_shares(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/filings")
def list_filings():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_filings(
			tenant_id=tenant,
			entity_id=request.args.get("entity_id"),
			filing_type=request.args.get("filing_type"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/filings")
def create_filing():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_filing(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/filings/<filing_id>/complete")
def complete_filing(filing_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().complete_filing(
			tenant, filing_id, body.get("reference_number", ""), body.get("filed_by_id", ""),
		)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/filings/<filing_id>")
def delete_filing(filing_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_filing(tenant, filing_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/resolutions")
def create_resolution():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_board_resolution(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/entities/<entity_id>/resolutions")
def list_resolutions(entity_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_board_resolutions(tenant, entity_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/dashboard")
def dashboard():
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().corporate_dashboard(tenant)))
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
