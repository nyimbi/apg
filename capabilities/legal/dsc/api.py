"""Document & eDiscovery — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import DocumentEDiscoveryService

_log = logging.getLogger(__name__)

bp = Blueprint("leg_dsc", __name__, url_prefix="/api/legal/dsc")
_svc: DocumentEDiscoveryService | None = None


def get_service() -> DocumentEDiscoveryService:
	global _svc
	if _svc is None:
		_svc = DocumentEDiscoveryService()
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


@bp.get("/documents")
def list_documents():
	tenant = request.args.get("tenant_id", "default")
	is_priv = request.args.get("is_privileged")
	on_hold = request.args.get("on_hold")
	try:
		items = _run(get_service().list_documents(
			tenant_id=tenant,
			matter_id=request.args.get("matter_id"),
			document_type=request.args.get("document_type"),
			is_privileged=None if is_priv is None else is_priv.lower() == "true",
			on_hold=None if on_hold is None else on_hold.lower() == "true",
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		_log.error("list_documents: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.get("/documents/<document_id>")
def get_document(document_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_document(tenant, document_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/documents")
def create_document():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_document(**body))), 201
	except Exception as exc:
		_log.error("create_document: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.put("/documents/<document_id>")
def update_document(document_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_document(tenant, document_id, **body)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/documents/<document_id>")
def delete_document(document_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_document(tenant, document_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/documents/search")
def search_documents():
	tenant = request.args.get("tenant_id", "default")
	query = request.args.get("q", "")
	try:
		items = _run(get_service().search_documents(tenant, query))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/privilege-log")
def log_privilege():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().log_privilege(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/privilege-log")
def list_privilege_log():
	tenant = request.args.get("tenant_id", "default")
	doc_id = request.args.get("document_id")
	try:
		items = _run(get_service().list_privilege_log(tenant, doc_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/holds")
def list_holds():
	tenant = request.args.get("tenant_id", "default")
	matter_id = request.args.get("matter_id")
	try:
		items = _run(get_service().list_litigation_holds(tenant, matter_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/holds/<hold_id>")
def get_hold(hold_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_litigation_hold(tenant, hold_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/holds")
def create_hold():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_litigation_hold(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/holds/<hold_id>/release")
def release_hold(hold_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().release_litigation_hold(tenant, hold_id, body.get("released_by", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/holds/<hold_id>")
def delete_hold(hold_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_litigation_hold(tenant, hold_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/productions")
def list_productions():
	tenant = request.args.get("tenant_id", "default")
	matter_id = request.args.get("matter_id")
	try:
		items = _run(get_service().list_production_sets(tenant, matter_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/productions")
def create_production():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_production_set(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/productions/<production_id>/finalize")
def finalize_production(production_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().finalize_production(tenant, production_id, body.get("finalized_by", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/stats")
def repo_stats():
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().repository_stats(tenant)))
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
