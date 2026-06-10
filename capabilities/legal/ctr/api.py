"""Contract Lifecycle Management — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import ContractLifecycleService

_log = logging.getLogger(__name__)

bp = Blueprint("leg_ctr", __name__, url_prefix="/api/legal/ctr")
_svc: ContractLifecycleService | None = None


def get_service() -> ContractLifecycleService:
	global _svc
	if _svc is None:
		_svc = ContractLifecycleService()
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


@bp.get("/contracts")
def list_contracts():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_contracts(
			tenant_id=tenant,
			status=request.args.get("status"),
			contract_type=request.args.get("contract_type"),
			counterparty_id=request.args.get("counterparty_id"),
			owner_id=request.args.get("owner_id"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		_log.error("list_contracts: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.get("/contracts/<contract_id>")
def get_contract(contract_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_contract(tenant, contract_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/contracts")
def create_contract():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_contract(**body))), 201
	except Exception as exc:
		_log.error("create_contract: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.put("/contracts/<contract_id>")
def update_contract(contract_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_contract(tenant, contract_id, **body)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/contracts/<contract_id>")
def delete_contract(contract_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_contract(tenant, contract_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/contracts/<contract_id>/submit")
def submit_for_review(contract_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().submit_for_review(tenant, contract_id, body.get("submitted_by", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/contracts/<contract_id>/execute")
def execute_contract(contract_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().execute_contract(tenant, contract_id, body.get("executed_by", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/contracts/<contract_id>/terminate")
def terminate_contract(contract_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().terminate_contract(
			tenant, contract_id, body.get("reason", ""), body.get("terminated_by", ""),
		)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/contracts/<contract_id>/redlines")
def list_redlines(contract_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_redlines(tenant, contract_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/redlines")
def create_redline():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_redline(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/redlines/<redline_id>/resolve")
def resolve_redline(redline_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().resolve_redline(
			tenant, redline_id, body.get("decision", ""), body.get("resolved_by_id", ""),
		)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/contracts/<contract_id>/obligations")
def list_obligations(contract_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_obligations(tenant, contract_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/obligations")
def create_obligation():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_obligation(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/approvals")
def create_approval():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_approval(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/approvals/<approval_id>/decide")
def decide_approval(approval_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().decide_approval(
			tenant, approval_id, body.get("decision", ""), body.get("comments", ""),
		)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/expiring")
def expiring_contracts():
	tenant = request.args.get("tenant_id", "default")
	days = int(request.args.get("days", 30))
	try:
		items = _run(get_service().list_expiring_contracts(tenant, days))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/dashboard")
def dashboard():
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().contract_dashboard(tenant)))
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
