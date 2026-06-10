"""GraphQL Federation Gateway — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import GraphQLGatewayService

_log = logging.getLogger(__name__)

bp = Blueprint("gql_gw", __name__, url_prefix="/api/gql")
_svc: GraphQLGatewayService | None = None


def _get_service() -> GraphQLGatewayService:
	global _svc
	if _svc is None:
		_svc = GraphQLGatewayService()
	return _svc


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as pool:
				return pool.submit(asyncio.run, coro).result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


@bp.get("/health")
def health():
	return jsonify(_run(_get_service().health_check())), 200


# ── GraphQL endpoint ─────────────────────────────────────────────

@bp.post("/graphql")
def graphql_execute():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = request.headers.get("X-Tenant-ID", body.get("tenant_id", "default"))
	user_id = request.headers.get("X-User-ID", "anonymous")
	query = body.get("query", "")
	variables = body.get("variables")
	operation_name = body.get("operationName")
	try:
		result = _run(svc.execute_query(tenant_id, query, variables, operation_name, user_id))
		status = 200 if not result.get("errors") else 400
		return jsonify(result), status
	except PermissionError as exc:
		return jsonify({"errors": [{"message": str(exc)}]}), 429
	except Exception as exc:
		_log.error("graphql_execute error: %s", exc)
		return jsonify({"errors": [{"message": "internal_error"}]}), 500


@bp.get("/graphql")
def graphql_introspect():
	svc = _get_service()
	tenant_id = request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))
	result = _run(svc.introspect(tenant_id))
	return jsonify({"data": result}), 200


# ── Subgraphs ─────────────────────────────────────────────────────

@bp.get("/subgraphs")
def list_subgraphs():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	enabled_raw = request.args.get("enabled")
	enabled = None if enabled_raw is None else (enabled_raw.lower() == "true")
	result = _run(svc.list_subgraphs(tenant_id, enabled=enabled))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/subgraphs")
def register_subgraph():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		result = _run(svc.register_subgraph(tenant_id=tenant_id, **body))
		return jsonify(result), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("register_subgraph error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/subgraphs/<name>")
def get_subgraph(name: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_subgraph(tenant_id, name))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/subgraphs/<name>")
def update_subgraph(name: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(svc.update_subgraph(tenant_id, name, **body))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/subgraphs/<name>")
def delete_subgraph(name: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.delete_subgraph(tenant_id, name))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/subgraphs/<name>/health")
def probe_subgraph(name: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.probe_subgraph_health(tenant_id, name))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/subgraphs/health/all")
def probe_all_subgraphs():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_run(svc.probe_all_subgraphs(tenant_id))), 200


# ── Schema ───────────────────────────────────────────────────────

@bp.get("/schema")
def compose_schema():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_run(svc.compose_schema(tenant_id))), 200


@bp.post("/schema/auto")
def auto_schema():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	semantic_model = body.get("semantic_model", {})
	return jsonify(_run(svc.auto_schema_from_semantic_model(tenant_id, semantic_model))), 200


@bp.post("/schema/flush")
def flush_schema_cache():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	return jsonify(_run(svc.flush_schema_cache(tenant_id))), 200


@bp.post("/schema/<name>/diff")
def schema_diff(name: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	new_sdl = body.get("sdl", "")
	try:
		return jsonify(_run(svc.get_schema_diff(tenant_id, name, new_sdl))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Persisted queries ─────────────────────────────────────────────

@bp.get("/persisted")
def list_persisted():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.list_persisted_queries(tenant_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/persisted")
def register_persisted():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(svc.register_persisted_query(tenant_id=tenant_id, **body))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/persisted/<query_id>/execute")
def execute_persisted(query_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	user_id = body.get("user_id", "anonymous")
	variables = body.get("variables")
	try:
		return jsonify(_run(svc.execute_persisted_query(tenant_id, query_id, variables, user_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/persisted/<query_id>")
def delete_persisted(query_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.delete_persisted_query(tenant_id, query_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── DataLoader ────────────────────────────────────────────────────

@bp.post("/dataloader/batch")
def dataloader_batch():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	loader_key = body.get("loader_key", "default")
	ids = body.get("ids", [])
	return jsonify(_run(svc.dataloader_batch(tenant_id, loader_key, ids))), 200


# ── Analytics ─────────────────────────────────────────────────────

@bp.get("/analytics")
def query_analytics():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_run(svc.query_analytics(tenant_id))), 200


@bp.get("/statistics")
def statistics():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_run(svc.gateway_statistics(tenant_id))), 200


@bp.get("/querylog")
def query_log():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	limit = int(request.args.get("limit", 100))
	result = _run(svc.get_query_log(tenant_id, limit))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/audit")
def audit_events():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.get_audit_events(tenant_id))
	return jsonify({"items": result, "total": len(result)}), 200
