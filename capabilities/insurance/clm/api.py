"""Flask Blueprint REST API for Claims Management (ins_clm)."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from flask import Blueprint, jsonify, request

from .service import ClaimsManagementService

_log = logging.getLogger(__name__)

clm_bp = Blueprint("ins_clm", __name__, url_prefix="/api/insurance/clm")
_svc = ClaimsManagementService()


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@clm_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@clm_bp.get("/describe")
def describe():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.describe(tenant)))


@clm_bp.get("/claims")
def list_claims():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	policy_id = request.args.get("policy_id")
	return jsonify(_run(_svc.list_claims(tenant, status, policy_id)))


@clm_bp.post("/claims")
def register_fnol():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.register_fnol(
			tenant_id=tenant,
			policy_id=data["policy_id"],
			policy_number=data["policy_number"],
			claimant_name=data["claimant_name"],
			claimant_id=data["claimant_id"],
			incident_date=data["incident_date"],
			incident_description=data["incident_description"],
			estimated_loss=Decimal(str(data["estimated_loss"])),
			reported_by=data.get("reported_by", ""),
			currency=data.get("currency", "KES"),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("register_fnol: %s", exc)
		return jsonify({"error": str(exc)}), 500


@clm_bp.get("/claims/<claim_id>")
def get_claim(claim_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_claim(tenant, claim_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@clm_bp.put("/claims/<claim_id>")
def update_claim(claim_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_claim(tenant, claim_id, data)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except (ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@clm_bp.delete("/claims/<claim_id>")
def delete_claim(claim_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_claim(tenant, claim_id, data.get("reason", "withdrawn"))))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@clm_bp.post("/claims/<claim_id>/reserve")
def set_reserve(claim_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.set_reserve(
			tenant_id=tenant,
			claim_id=claim_id,
			reserve_amount=Decimal(str(data["reserve_amount"])),
			reserve_type=data.get("reserve_type", "outstanding"),
			set_by=data.get("set_by", ""),
			justification=data.get("justification", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@clm_bp.post("/claims/<claim_id>/payment")
def process_payment(claim_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.process_payment(
			tenant_id=tenant,
			claim_id=claim_id,
			payment_amount=Decimal(str(data["payment_amount"])),
			payment_type=data.get("payment_type", "partial"),
			payee_name=data["payee_name"],
			payee_account=data["payee_account"],
			payment_reference=data["payment_reference"],
			authorised_by=data.get("authorised_by", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@clm_bp.post("/claims/<claim_id>/fraud")
def assess_fraud(claim_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.assess_fraud_risk(
			tenant_id=tenant,
			claim_id=claim_id,
			fraud_score=float(data.get("fraud_score", 0.0)),
			indicators=data.get("indicators", []),
			assessed_by=data.get("assessed_by", ""),
			recommendation=data.get("recommendation", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@clm_bp.post("/claims/<claim_id>/approve")
def approve_claim(claim_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.approve_claim(
			tenant_id=tenant,
			claim_id=claim_id,
			approved_amount=Decimal(str(data["approved_amount"])),
			approved_by=data.get("approved_by", ""),
		))
		return jsonify(rec), 200
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@clm_bp.post("/claims/<claim_id>/subrogation")
def initiate_subrogation(claim_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.initiate_subrogation(
			tenant_id=tenant,
			claim_id=claim_id,
			third_party_name=data["third_party_name"],
			third_party_id=data["third_party_id"],
			recovery_amount=Decimal(str(data["recovery_amount"])),
			legal_reference=data.get("legal_reference"),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@clm_bp.get("/summary")
def claims_summary():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.claims_summary(tenant)))


@clm_bp.get("/audit")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant)))
