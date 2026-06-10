"""Data Catalog — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import DataCatalogService

_log = logging.getLogger(__name__)

bp = Blueprint("dcat_cat", __name__, url_prefix="/api/dcat/cat")
_svc: DataCatalogService | None = None


def _get_service() -> DataCatalogService:
	global _svc
	if _svc is None:
		_svc = DataCatalogService()
	return _svc


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as pool:
				future = pool.submit(asyncio.run, coro)
				return future.result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


@bp.get("/health")
def health():
	result = _run(_get_service().health_check())
	return jsonify(result), 200


@bp.get("/datasets")
def list_datasets():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	owner = request.args.get("owner")
	domain = request.args.get("domain")
	classification = request.args.get("classification")
	tags_raw = request.args.get("tags")
	tags = tags_raw.split(",") if tags_raw else None
	result = _run(svc.list_datasets(
		tenant_id=tenant_id,
		owner=owner,
		domain=domain,
		classification=classification,
		tags=tags,
	))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/datasets")
def create_dataset():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		result = _run(svc.create_dataset(tenant_id=tenant_id, **body))
		return jsonify(result), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("create_dataset error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/datasets/<dataset_id>")
def get_dataset(dataset_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		result = _run(svc.get_dataset(tenant_id, dataset_id))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/datasets/<dataset_id>")
def update_dataset(dataset_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		result = _run(svc.update_dataset(tenant_id, dataset_id, **body))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("update_dataset error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.delete("/datasets/<dataset_id>")
def delete_dataset(dataset_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		result = _run(svc.delete_dataset(tenant_id, dataset_id))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/datasets/search")
def search_datasets():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	query = request.args.get("q", "")
	result = _run(svc.search_datasets(tenant_id, query))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/lineage")
def add_lineage():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		result = _run(svc.add_lineage_edge(tenant_id=tenant_id, **body))
		return jsonify(result), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/lineage")
def list_lineage():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.list_lineage_edges(tenant_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/lineage/<dataset_id>/upstream")
def lineage_upstream(dataset_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	depth = int(request.args.get("depth", 5))
	result = _run(svc.get_lineage_upstream(tenant_id, dataset_id, depth))
	return jsonify(result), 200


@bp.get("/lineage/<dataset_id>/downstream")
def lineage_downstream(dataset_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	depth = int(request.args.get("depth", 5))
	result = _run(svc.get_lineage_downstream(tenant_id, dataset_id, depth))
	return jsonify(result), 200


@bp.post("/glossary")
def create_glossary_term():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		result = _run(svc.create_glossary_term(tenant_id=tenant_id, **body))
		return jsonify(result), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/glossary")
def list_glossary():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	domain = request.args.get("domain")
	result = _run(svc.list_glossary_terms(tenant_id, domain))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/glossary/<term_id>")
def get_glossary_term(term_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		result = _run(svc.get_glossary_term(tenant_id, term_id))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/glossary/<term_id>")
def delete_glossary_term(term_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		result = _run(svc.delete_glossary_term(tenant_id, term_id))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/tags")
def create_tag():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	result = _run(svc.create_tag(tenant_id=tenant_id, **body))
	return jsonify(result), 201


@bp.get("/tags")
def list_tags():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.list_tags(tenant_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/statistics")
def statistics():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.catalog_statistics(tenant_id))
	return jsonify(result), 200


@bp.get("/audit")
def audit_events():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.get_audit_events(tenant_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/atlas/entity/<dataset_id>")
def atlas_entity(dataset_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		result = _run(svc.atlas_get_entity(tenant_id, dataset_id))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/impact/<dataset_id>")
def impact_analysis(dataset_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.get_impact_analysis(tenant_id, dataset_id))
	return jsonify(result), 200
