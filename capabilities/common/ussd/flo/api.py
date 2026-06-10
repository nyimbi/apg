"""Flask Blueprint REST API for ussd_flo capability."""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import UssdFloService

_log = logging.getLogger(__name__)

bp = Blueprint("ussd_flo", __name__, url_prefix="/api/ussd/flo")
_svc: UssdFloService | None = None


def get_service() -> UssdFloService:
	global _svc
	if _svc is None:
		_svc = UssdFloService()
	return _svc


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				fut = pool.submit(asyncio.run, coro)
				return fut.result()
		return loop.run_until_complete(coro)
	except Exception:
		return asyncio.run(coro)


# ── Health ────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
	return jsonify(_run(get_service().health_check())), 200


# ── Flow endpoints ────────────────────────────────────────────────────────────

@bp.get("/flows")
def list_flows():
	result = _run(get_service().list_flows(
		tenant_id=_tenant(),
		service_code=request.args.get("service_code"),
		status=request.args.get("status"),
		tag=request.args.get("tag"),
	))
	return jsonify(result), 200


@bp.post("/flows")
def create_flow():
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().create_flow(
			name=data["name"],
			service_code=data["service_code"],
			root_node_id=data["root_node_id"],
			tenant_id=_tenant(),
			description=data.get("description", ""),
			languages=data.get("languages"),
			tags=data.get("tags"),
			metadata=data.get("metadata"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("create_flow error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/flows/<flow_id>")
def get_flow(flow_id: str):
	try:
		return jsonify(_run(get_service().get_flow(flow_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("get_flow error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.put("/flows/<flow_id>")
def update_flow(flow_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().update_flow(
			flow_id=flow_id, tenant_id=_tenant(),
			name=data.get("name"), description=data.get("description"),
			root_node_id=data.get("root_node_id"), languages=data.get("languages"),
			tags=data.get("tags"), status=data.get("status"), metadata=data.get("metadata"),
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("update_flow error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.delete("/flows/<flow_id>")
def delete_flow(flow_id: str):
	try:
		return jsonify(_run(get_service().delete_flow(flow_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("delete_flow error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.post("/flows/<flow_id>/activate")
def activate_flow(flow_id: str):
	try:
		return jsonify(_run(get_service().activate_flow(flow_id, _tenant()))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("activate_flow error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.post("/flows/<flow_id>/archive")
def archive_flow(flow_id: str):
	try:
		return jsonify(_run(get_service().archive_flow(flow_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("archive_flow error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/flows/<flow_id>/validate")
def validate_flow(flow_id: str):
	try:
		return jsonify(_run(get_service().validate_flow(flow_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("validate_flow error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.post("/flows/<flow_id>/export")
def export_flow(flow_id: str):
	try:
		return jsonify(_run(get_service().export_flow(flow_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("export_flow error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.post("/flows/import")
def import_flow():
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().import_flow(data, tenant_id=_tenant(), overwrite=data.get("overwrite", False)))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("import_flow error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


# ── Node endpoints ────────────────────────────────────────────────────────────

@bp.get("/flows/<flow_id>/nodes")
def list_nodes(flow_id: str):
	try:
		return jsonify(_run(get_service().list_nodes(flow_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("list_nodes error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.post("/flows/<flow_id>/nodes")
def add_node(flow_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().add_node(
			flow_id=flow_id, node_id=data["node_id"], node_type=data["node_type"],
			title=data["title"], tenant_id=_tenant(), body=data.get("body", ""),
			items=data.get("items"), position_x=data.get("position_x", 0.0),
			position_y=data.get("position_y", 0.0), metadata=data.get("metadata"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("add_node error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/flows/<flow_id>/nodes/<node_id>")
def get_node(flow_id: str, node_id: str):
	try:
		return jsonify(_run(get_service().get_node(flow_id, node_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("get_node error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.put("/flows/<flow_id>/nodes/<node_id>")
def update_node(flow_id: str, node_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().update_node(
			flow_id=flow_id, node_id=node_id, tenant_id=_tenant(),
			title=data.get("title"), body=data.get("body"), items=data.get("items"),
			position_x=data.get("position_x"), position_y=data.get("position_y"),
			metadata=data.get("metadata"),
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("update_node error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.delete("/flows/<flow_id>/nodes/<node_id>")
def delete_node(flow_id: str, node_id: str):
	try:
		return jsonify(_run(get_service().delete_node(flow_id, node_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("delete_node error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


# ── Edge endpoints ────────────────────────────────────────────────────────────

@bp.get("/flows/<flow_id>/edges")
def list_edges(flow_id: str):
	try:
		return jsonify(_run(get_service().list_edges(flow_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("list_edges error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.post("/flows/<flow_id>/edges")
def add_edge(flow_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().add_edge(
			flow_id=flow_id, source_node_id=data["source_node_id"],
			target_node_id=data["target_node_id"], tenant_id=_tenant(),
			label=data.get("label", ""), condition=data.get("condition"),
			priority=data.get("priority", 0), metadata=data.get("metadata"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("add_edge error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.delete("/flows/<flow_id>/edges/<edge_id>")
def delete_edge(flow_id: str, edge_id: str):
	try:
		return jsonify(_run(get_service().delete_edge(edge_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("delete_edge error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


# ── Conditional routing ───────────────────────────────────────────────────────

@bp.post("/flows/<flow_id>/route")
def resolve_next_node(flow_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().resolve_next_node(
			flow_id=flow_id, current_node_id=data["current_node_id"],
			context=data.get("context", {}), tenant_id=_tenant(),
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("resolve_next_node error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


# ── Translation endpoints ─────────────────────────────────────────────────────

@bp.get("/flows/<flow_id>/translations")
def list_translations(flow_id: str):
	try:
		return jsonify(_run(get_service().list_translations(flow_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("list_translations error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.post("/flows/<flow_id>/translations")
def add_translation(flow_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().add_translation(
			flow_id=flow_id, language=data["language"],
			translations=data["translations"], tenant_id=_tenant(),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("add_translation error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/flows/<flow_id>/translations/<language>")
def get_translation(flow_id: str, language: str):
	try:
		return jsonify(_run(get_service().get_translation(flow_id, language, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("get_translation error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.delete("/flows/<flow_id>/translations/<language>")
def delete_translation(flow_id: str, language: str):
	try:
		return jsonify(_run(get_service().delete_translation(flow_id, language, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("delete_translation error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


# ── Version endpoints ─────────────────────────────────────────────────────────

@bp.get("/flows/<flow_id>/versions")
def list_versions(flow_id: str):
	try:
		return jsonify(_run(get_service().list_flow_versions(flow_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("list_versions error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.post("/flows/<flow_id>/versions")
def snapshot_flow(flow_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().snapshot_flow(flow_id, label=data.get("label", "manual"), tenant_id=_tenant()))
		return jsonify(result), 201
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("snapshot_flow error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.post("/flows/<flow_id>/versions/<version_id>/restore")
def restore_version(flow_id: str, version_id: str):
	try:
		return jsonify(_run(get_service().restore_flow_version(flow_id, version_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("restore_version error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


# ── A/B test endpoints ────────────────────────────────────────────────────────

@bp.get("/abtests")
def list_ab_tests():
	result = _run(get_service().list_ab_tests(tenant_id=_tenant(), status=request.args.get("status")))
	return jsonify(result), 200


@bp.post("/abtests")
def create_ab_test():
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().create_ab_test(
			name=data["name"], service_code=data["service_code"],
			control_flow_id=data["control_flow_id"], variant_flow_id=data["variant_flow_id"],
			tenant_id=_tenant(), split_percentage=data.get("split_percentage", 50.0),
			metadata=data.get("metadata"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("create_ab_test error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/abtests/<test_id>")
def get_ab_test(test_id: str):
	try:
		return jsonify(_run(get_service().get_ab_test(test_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("get_ab_test error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.put("/abtests/<test_id>")
def update_ab_test(test_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(get_service().update_ab_test(
			test_id=test_id, tenant_id=_tenant(),
			split_percentage=data.get("split_percentage"),
			status=data.get("status"), metadata=data.get("metadata"),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("update_ab_test error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.delete("/abtests/<test_id>")
def delete_ab_test(test_id: str):
	try:
		return jsonify(_run(get_service().delete_ab_test(test_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("delete_ab_test error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/abtests/<test_id>/results")
def ab_test_results(test_id: str):
	try:
		return jsonify(_run(get_service().get_ab_test_results(test_id, _tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("ab_test_results error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


# ── Analytics / dashboard ─────────────────────────────────────────────────────

@bp.get("/dashboard")
def dashboard():
	return jsonify(_run(get_service().dashboard_summary(tenant_id=_tenant()))), 200


@bp.get("/audit")
def audit_events():
	limit = int(request.args.get("limit", 100))
	return jsonify(_run(get_service().get_audit_events(tenant_id=_tenant(), limit=limit))), 200
