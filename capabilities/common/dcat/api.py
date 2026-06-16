# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025

from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Any

from flask import Blueprint, jsonify, request

from .models import Dataset, DatasetSearch, DatasetTag, LineageEdge
from .service import DataCatalogService

log = logging.getLogger(__name__)

bp = Blueprint("dcat", __name__, url_prefix="/api/common/dcat")

# Module-level service singleton — replace with app-context factory if needed
_svc = DataCatalogService()


def _run(coro):
	"""Run an async coroutine from a synchronous Flask view."""
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


def _tenant() -> str:
	"""Extract tenant_id from request headers or query params."""
	return (
		request.headers.get("X-Tenant-ID")
		or request.args.get("tenant_id")
		or "default"
	)


def _json_error(msg: str, status: int = 400) -> tuple[Any, int]:
	return jsonify({"error": msg}), status


# ------------------------------------------------------------------
# POST /datasets — register a new dataset
# ------------------------------------------------------------------

@bp.route("/datasets", methods=["POST"])
def create_dataset():
	data = request.get_json(silent=True)
	if not data:
		return _json_error("Request body must be JSON")
	try:
		data.setdefault("tenant_id", _tenant())
		ds = Dataset(**data)
	except Exception as exc:
		return _json_error(f"Validation error: {exc}")
	try:
		dataset_id = _run(_svc.register_dataset(ds))
	except Exception as exc:
		log.exception("dcat.create_dataset failed")
		return _json_error(f"Internal error: {exc}", 500)
	return jsonify({"id": dataset_id}), 201


# ------------------------------------------------------------------
# GET /datasets — list / search datasets
# ------------------------------------------------------------------

@bp.route("/datasets", methods=["GET"])
def list_datasets():
	tenant_id = _tenant()
	try:
		q = DatasetSearch(
			tenant_id=tenant_id,
			query=request.args.get("q"),
			limit=int(request.args.get("limit", 50)),
			offset=int(request.args.get("offset", 0)),
		)
	except Exception as exc:
		return _json_error(f"Validation error: {exc}")
	results = _run(_svc.search_datasets(q))
	return jsonify([ds.model_dump(mode="json") for ds in results])


# ------------------------------------------------------------------
# GET /datasets/<id> — fetch a single dataset
# ------------------------------------------------------------------

@bp.route("/datasets/<dataset_id>", methods=["GET"])
def get_dataset(dataset_id: str):
	tenant_id = _tenant()
	ds = _run(_svc.get_dataset(tenant_id, dataset_id))
	if ds is None:
		return _json_error(f"Dataset {dataset_id!r} not found", 404)
	return jsonify(ds.model_dump(mode="json"))


# ------------------------------------------------------------------
# POST /datasets/<id>/tags — add tags to a dataset
# ------------------------------------------------------------------

@bp.route("/datasets/<dataset_id>/tags", methods=["POST"])
def add_tags(dataset_id: str):
	tenant_id = _tenant()
	data = request.get_json(silent=True)
	if not isinstance(data, list):
		return _json_error("Body must be a JSON array of tag objects")
	try:
		tags = [DatasetTag(**t) for t in data]
	except Exception as exc:
		return _json_error(f"Validation error: {exc}")
	try:
		_run(_svc.tag_dataset(dataset_id, tenant_id, tags))
	except KeyError as exc:
		return _json_error(str(exc), 404)
	except Exception as exc:
		log.exception("dcat.add_tags failed")
		return _json_error(f"Internal error: {exc}", 500)
	return jsonify({"tagged": len(tags)}), 200


# ------------------------------------------------------------------
# GET /datasets/<id>/lineage — fetch lineage graph
# ------------------------------------------------------------------

@bp.route("/datasets/<dataset_id>/lineage", methods=["GET"])
def get_lineage(dataset_id: str):
	tenant_id = _tenant()
	try:
		depth = int(request.args.get("depth", 3))
	except ValueError:
		return _json_error("depth must be an integer")
	try:
		graph = _run(_svc.get_lineage(dataset_id, tenant_id, depth=depth))
	except Exception as exc:
		log.exception("dcat.get_lineage failed")
		return _json_error(f"Internal error: {exc}", 500)
	return jsonify(graph)


# ------------------------------------------------------------------
# POST /lineage — add a lineage edge
# ------------------------------------------------------------------

@bp.route("/lineage", methods=["POST"])
def add_lineage():
	data = request.get_json(silent=True)
	if not data:
		return _json_error("Request body must be JSON")
	try:
		data.setdefault("tenant_id", _tenant())
		edge = LineageEdge(**data)
	except Exception as exc:
		return _json_error(f"Validation error: {exc}")
	try:
		_run(_svc.add_lineage(edge))
	except Exception as exc:
		log.exception("dcat.add_lineage failed")
		return _json_error(f"Internal error: {exc}", 500)
	return jsonify({"id": edge.id}), 201


# ------------------------------------------------------------------
# GET /search — full-text + structured search
# ------------------------------------------------------------------

@bp.route("/search", methods=["GET"])
def search():
	tenant_id = _tenant()
	try:
		q = DatasetSearch(
			tenant_id=tenant_id,
			query=request.args.get("q"),
			owner=request.args.get("owner"),
			tag_key=request.args.get("tag_key"),
			tag_value=request.args.get("tag_value"),
			classification=request.args.get("classification"),
			limit=int(request.args.get("limit", 50)),
			offset=int(request.args.get("offset", 0)),
		)
	except Exception as exc:
		return _json_error(f"Validation error: {exc}")
	results = _run(_svc.search_datasets(q))
	return jsonify([ds.model_dump(mode="json") for ds in results])


# ------------------------------------------------------------------
# GET /datasets/<id>/quality — compute quality score
# ------------------------------------------------------------------

@bp.route("/datasets/<dataset_id>/quality", methods=["GET"])
def quality_score(dataset_id: str):
	tenant_id = _tenant()
	try:
		score = _run(_svc.score_quality(dataset_id, tenant_id))
	except KeyError as exc:
		return _json_error(str(exc), 404)
	except Exception as exc:
		log.exception("dcat.quality_score failed")
		return _json_error(f"Internal error: {exc}", 500)
	return jsonify(score.model_dump(mode="json"))
