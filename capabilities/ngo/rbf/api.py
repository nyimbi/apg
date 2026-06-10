"""Results-Based Financing — Flask Blueprint with async REST endpoints."""
from __future__ import annotations

import logging
from decimal import Decimal

from flask import Blueprint, jsonify, request

from .service import ResultsBasedFinancingService

_log = logging.getLogger(__name__)

bp = Blueprint("ngo_rbf", __name__, url_prefix="/api/ngo/rbf")

_svc: ResultsBasedFinancingService | None = None


def _get_service() -> ResultsBasedFinancingService:
	global _svc
	if _svc is None:
		_svc = ResultsBasedFinancingService()
	return _svc


def _run(coro):
	import asyncio
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as pool:
				return pool.submit(asyncio.run, coro).result()
		return loop.run_until_complete(coro)
	except Exception as exc:
		_log.error("async execution error: %s", exc)
		raise


@bp.get("/health")
def health():
	return jsonify(_run(_get_service().health_check())), 200


# ── contracts ──────────────────────────────────────────────────────────────────

@bp.get("/contracts")
def list_contracts():
	result = _run(_get_service().list_contracts(
		status=request.args.get("status"),
		programme_id=request.args.get("programme_id"),
	))
	return jsonify({"contracts": result, "count": len(result)}), 200


@bp.get("/contracts/<contract_id>")
def get_contract(contract_id: str):
	try:
		return jsonify(_run(_get_service().get_contract(contract_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/contracts")
def create_contract():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_contract(
			programme_id=data["programme_id"],
			funder_reference=data["funder_reference"],
			title=data["title"],
			total_value=Decimal(str(data["total_value"])),
			start_date=data["start_date"],
			end_date=data["end_date"],
			description=data.get("description", ""),
			currency=data.get("currency", "KES"),
			payment_model=data.get("payment_model", "output_based"),
			contract_manager=data.get("contract_manager", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/contracts/<contract_id>")
def update_contract(contract_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_get_service().update_contract(contract_id, **data))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/contracts/<contract_id>")
def delete_contract(contract_id: str):
	try:
		return jsonify(_run(_get_service().delete_contract(contract_id))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/contracts/<contract_id>/activate")
def activate_contract(contract_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_get_service().activate_contract(contract_id, approved_by=data.get("approved_by", "")))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/contracts/<contract_id>/performance")
def contract_performance(contract_id: str):
	try:
		return jsonify(_run(_get_service().contract_performance_summary(contract_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/contracts/<contract_id>/dli-achievement")
def dli_achievement(contract_id: str):
	try:
		result = _run(_get_service().dli_achievement_report(contract_id))
		return jsonify({"dlis": result, "count": len(result)}), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── DLIs ───────────────────────────────────────────────────────────────────────

@bp.get("/dlis")
def list_dlis():
	result = _run(_get_service().list_dlis(contract_id=request.args.get("contract_id")))
	return jsonify({"dlis": result, "count": len(result)}), 200


@bp.get("/dlis/<dli_id>")
def get_dli(dli_id: str):
	try:
		return jsonify(_run(_get_service().get_dli(dli_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/dlis")
def create_dli():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_dli(
			contract_id=data["contract_id"],
			name=data["name"],
			target_value=float(data["target_value"]),
			price_per_unit=Decimal(str(data["price_per_unit"])),
			due_date=data["due_date"],
			description=data.get("description", ""),
			indicator_code=data.get("indicator_code", ""),
			unit=data.get("unit", ""),
			currency=data.get("currency", "KES"),
			verification_method=data.get("verification_method", "third_party"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── claims ─────────────────────────────────────────────────────────────────────

@bp.get("/claims")
def list_claims():
	result = _run(_get_service().list_claims(
		contract_id=request.args.get("contract_id"),
		status=request.args.get("status"),
	))
	return jsonify({"claims": result, "count": len(result)}), 200


@bp.get("/claims/<claim_id>")
def get_claim(claim_id: str):
	try:
		return jsonify(_run(_get_service().get_claim(claim_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/claims")
def submit_claim():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().submit_result_claim(
			contract_id=data["contract_id"],
			dli_id=data["dli_id"],
			claimed_value=float(data["claimed_value"]),
			claim_date=data["claim_date"],
			submitted_by=data["submitted_by"],
			evidence_references=data.get("evidence_references", []),
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── verifications ──────────────────────────────────────────────────────────────

@bp.get("/verifications")
def list_verifications():
	result = _run(_get_service().list_verifications(claim_id=request.args.get("claim_id")))
	return jsonify({"verifications": result, "count": len(result)}), 200


@bp.post("/verifications")
def create_verification():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_verification(
			claim_id=data["claim_id"],
			verifier=data["verifier"],
			verification_date=data["verification_date"],
			verified_value=float(data.get("verified_value", 0)),
			accepted=bool(data.get("accepted", True)),
			methodology=data.get("methodology", ""),
			findings=data.get("findings", ""),
			adjustments=data.get("adjustments", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── payment triggers ───────────────────────────────────────────────────────────

@bp.get("/payment-triggers")
def list_payment_triggers():
	result = _run(_get_service().list_payment_triggers(contract_id=request.args.get("contract_id")))
	return jsonify({"payment_triggers": result, "count": len(result)}), 200


@bp.post("/payment-triggers")
def trigger_payment():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().trigger_payment(
			contract_id=data["contract_id"],
			claim_id=data["claim_id"],
			verification_id=data["verification_id"],
			amount=Decimal(str(data["amount"])),
			payment_date=data["payment_date"],
			approved_by=data["approved_by"],
			reference=data["reference"],
			currency=data.get("currency", "KES"),
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/payment-triggers/<trigger_id>/confirm")
def confirm_payment(trigger_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_get_service().confirm_payment(trigger_id, confirmed_by=data.get("confirmed_by", "")))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── portfolio ──────────────────────────────────────────────────────────────────

@bp.get("/portfolio/summary")
def portfolio_summary():
	return jsonify(_run(_get_service().portfolio_rbf_summary())), 200


@bp.get("/audit-events")
def get_audit_events():
	limit = int(request.args.get("limit", 100))
	result = _run(_get_service().get_audit_events(limit=limit))
	return jsonify({"events": result, "count": len(result)}), 200
