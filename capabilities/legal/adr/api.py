"""ADR / Dispute Resolution — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import ADRDisputeResolutionService

_log = logging.getLogger(__name__)

bp = Blueprint("leg_adr", __name__, url_prefix="/api/legal/adr")
_svc: ADRDisputeResolutionService | None = None


def get_service() -> ADRDisputeResolutionService:
	global _svc
	if _svc is None:
		_svc = ADRDisputeResolutionService()
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


@bp.get("/cases")
def list_cases():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_cases(
			tenant_id=tenant,
			case_type=request.args.get("case_type"),
			status=request.args.get("status"),
			claimant_id=request.args.get("claimant_id"),
			respondent_id=request.args.get("respondent_id"),
			seat=request.args.get("seat"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/cases/<case_id>")
def get_case(case_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_case(tenant, case_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/cases")
def create_case():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_case(**body))), 201
	except Exception as exc:
		_log.error("create_case: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.put("/cases/<case_id>")
def update_case(case_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_case(tenant, case_id, **body)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/cases/<case_id>")
def delete_case(case_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_case(tenant, case_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/cases/<case_id>/advance")
def advance_status(case_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().advance_case_status(
			tenant, case_id, body.get("new_status", ""), body.get("notes", ""),
		)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/cases/<case_id>/neutrals")
def list_neutrals(case_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_neutrals(tenant, case_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/neutrals")
def appoint_neutral():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().appoint_neutral(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/neutrals/<neutral_id>/challenge")
def challenge_neutral(neutral_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().challenge_neutral(
			tenant, neutral_id, body.get("reason", ""), body.get("challenged_by", ""),
		)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/neutrals/<neutral_id>")
def remove_neutral(neutral_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", request.args.get("tenant_id", "default"))
	try:
		return jsonify(_run(get_service().remove_neutral(tenant, neutral_id, body.get("reason", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/cases/<case_id>/proceedings")
def list_proceedings(case_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_proceedings(tenant, case_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/proceedings")
def create_proceeding():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_proceeding(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/proceedings/<proceeding_id>/conclude")
def conclude_proceeding(proceeding_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().conclude_proceeding(
			tenant, proceeding_id,
			body.get("actual_date", ""),
			body.get("minutes_reference", ""),
			body.get("duration_hours", 0.0),
		)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/proceedings/<proceeding_id>")
def delete_proceeding(proceeding_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_proceeding(tenant, proceeding_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/awards")
def list_awards():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_awards(tenant, request.args.get("case_id")))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/awards/<award_id>")
def get_award(award_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_award(tenant, award_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/awards")
def create_award():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_award(**body))), 201
	except Exception as exc:
		_log.error("create_award: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.put("/awards/<award_id>")
def update_award(award_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_award(tenant, award_id, **body)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/awards/<award_id>")
def delete_award(award_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_award(tenant, award_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/awards/<award_id>/challenge")
def challenge_award(award_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().challenge_award(
			tenant, award_id, body.get("basis", ""), body.get("filed_by", ""),
		)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/awards/<award_id>/enforce")
def enforce_award(award_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().file_enforcement_action(**{"award_id": award_id, **body}))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/settlements")
def list_settlements():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_settlements(tenant, request.args.get("case_id")))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/settlements")
def create_settlement():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_settlement(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/settlements/<settlement_id>")
def update_settlement(settlement_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_settlement(tenant, settlement_id, **body)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/settlements/<settlement_id>")
def delete_settlement(settlement_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_settlement(tenant, settlement_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/dashboard")
def dashboard():
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().adr_dashboard(tenant)))
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
