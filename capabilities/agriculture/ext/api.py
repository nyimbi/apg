"""Extension Services Flask Blueprint — agr_ext."""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request
from .service import ExtensionServicesService

_log = logging.getLogger(__name__)
bp = Blueprint("agr_ext", __name__, url_prefix="/api/agriculture/ext")
_svc: dict[str, ExtensionServicesService] = {}


def _get_svc(t: str = "default") -> ExtensionServicesService:
	if t not in _svc:
		_svc[t] = ExtensionServicesService(tenant_id=t)
	return _svc[t]


def _t() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@bp.get("/health")
async def health():
	return jsonify(await _get_svc(_t()).health_check()), 200


@bp.get("/advisories")
async def list_advisories():
	svc = _get_svc(_t())
	fu_str = request.args.get("follow_up_required")
	fu = None if fu_str is None else fu_str.lower() == "true"
	items = await svc.list_advisories(
		farmer_id=request.args.get("farmer_id"),
		extension_worker_id=request.args.get("extension_worker_id"),
		channel=request.args.get("channel"),
		follow_up_required=fu,
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/advisories")
async def create_advisory():
	try:
		return jsonify(await _get_svc(_t()).create_advisory(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/advisories/<advisory_id>")
async def get_advisory(advisory_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_advisory(advisory_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/advisories/<advisory_id>/follow-up")
async def follow_up(advisory_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _get_svc(_t()).mark_follow_up_done(advisory_id, body.get("notes"))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/advisories/<advisory_id>")
async def delete_advisory(advisory_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_advisory(advisory_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/demo-plots")
async def list_demo_plots():
	svc = _get_svc(_t())
	items = await svc.list_demo_plots(
		extension_worker_id=request.args.get("extension_worker_id"),
		crop_type=request.args.get("crop_type"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/demo-plots")
async def create_demo_plot():
	try:
		return jsonify(await _get_svc(_t()).create_demo_plot(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/demo-plots/<plot_id>")
async def update_demo_plot(plot_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_demo_plot(plot_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/demo-plots/<plot_id>")
async def delete_demo_plot(plot_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_demo_plot(plot_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/trainings")
async def list_trainings():
	svc = _get_svc(_t())
	items = await svc.list_trainings(trainer_id=request.args.get("trainer_id"), status=request.args.get("status"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/trainings")
async def create_training():
	try:
		return jsonify(await _get_svc(_t()).create_training(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/trainings/<training_id>")
async def update_training(training_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_training(training_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/trainings/<training_id>")
async def delete_training(training_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_training(training_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/knowledge")
async def list_knowledge():
	svc = _get_svc(_t())
	items = await svc.list_knowledge(
		category=request.args.get("category"),
		crop_type=request.args.get("crop_type"),
		language=request.args.get("language"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.get("/knowledge/search")
async def search_knowledge():
	svc = _get_svc(_t())
	results = await svc.search_knowledge(
		query=request.args.get("q", ""),
		language=request.args.get("language"),
	)
	return jsonify({"results": results, "count": len(results)}), 200


@bp.post("/knowledge")
async def create_knowledge():
	try:
		return jsonify(await _get_svc(_t()).create_knowledge_article(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/knowledge/<article_id>")
async def get_knowledge(article_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_knowledge_article(article_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/knowledge/<article_id>")
async def update_knowledge(article_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_knowledge_article(article_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/knowledge/<article_id>")
async def delete_knowledge(article_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_knowledge_article(article_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/summary")
async def reach_summary():
	return jsonify(await _get_svc(_t()).get_extension_reach_summary()), 200


@bp.get("/audit")
async def get_audit():
	events = await _get_svc(_t()).get_audit_events(int(request.args.get("limit", 100)))
	return jsonify({"events": events, "count": len(events)}), 200
