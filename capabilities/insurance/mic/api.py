"""Flask Blueprint REST API for Micro-Insurance Platform (ins_mic)."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from flask import Blueprint, jsonify, request

from .service import MicroInsurancePlatformService

_log = logging.getLogger(__name__)

mic_bp = Blueprint("ins_mic", __name__, url_prefix="/api/insurance/mic")
_svc = MicroInsurancePlatformService()


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@mic_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@mic_bp.get("/describe")
def describe():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.describe(tenant)))


@mic_bp.get("/products")
def list_products():
	tenant = request.args.get("tenant_id", "default")
	product_type = request.args.get("product_type")
	return jsonify(_run(_svc.list_products(tenant, product_type)))


@mic_bp.post("/products")
def create_product():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.create_product(
			tenant_id=tenant,
			product_code=data["product_code"],
			product_name=data["product_name"],
			product_type=data["product_type"],
			sum_insured=Decimal(str(data["sum_insured"])),
			premium=Decimal(str(data["premium"])),
			coverage_days=int(data["coverage_days"]),
			ussd_menu_code=data.get("ussd_menu_code", "*384#"),
			airtime_deduction=bool(data.get("airtime_deduction", False)),
			mobile_money_payout=bool(data.get("mobile_money_payout", True)),
			currency=data.get("currency", "KES"),
			description=data.get("description", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@mic_bp.get("/products/<product_id>")
def get_product(product_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_product(tenant, product_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@mic_bp.put("/products/<product_id>")
def update_product(product_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_product(tenant, product_id, data)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@mic_bp.delete("/products/<product_id>")
def delete_product(product_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_product(tenant, product_id)))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@mic_bp.post("/ussd")
def process_ussd():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.process_ussd_session(
			tenant_id=tenant,
			session_id=data["session_id"],
			msisdn=data["msisdn"],
			service_code=data.get("service_code", "*384#"),
			input_text=data.get("input_text", ""),
			step=int(data.get("step", 0)),
		))
		return jsonify(rec)
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@mic_bp.post("/enrolments")
def enrol_subscriber():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.enrol_subscriber(
			tenant_id=tenant,
			msisdn=data["msisdn"],
			product_code=data["product_code"],
			name=data["name"],
			id_number=data.get("id_number"),
			enrolment_channel=data.get("enrolment_channel", "ussd"),
			payment_method=data.get("payment_method", "airtime"),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@mic_bp.get("/enrolments")
def list_enrolments():
	tenant = request.args.get("tenant_id", "default")
	product_code = request.args.get("product_code")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_enrolments(tenant, product_code, status)))


@mic_bp.get("/enrolments/<enrolment_id>")
def get_enrolment(enrolment_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_enrolment(tenant, enrolment_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@mic_bp.put("/enrolments/<enrolment_id>")
def update_enrolment(enrolment_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_enrolment(tenant, enrolment_id, data)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@mic_bp.delete("/enrolments/<enrolment_id>")
def cancel_enrolment(enrolment_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.cancel_enrolment(tenant, enrolment_id, data.get("reason", "cancelled"))))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@mic_bp.post("/airtime/deduct")
def deduct_airtime():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.deduct_airtime_premium(
			tenant_id=tenant,
			msisdn=data["msisdn"],
			product_code=data["product_code"],
			amount=Decimal(str(data["amount"])),
			operator=data["operator"],
			deduction_reference=data["deduction_reference"],
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@mic_bp.post("/claims")
def register_claim():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.register_claim(
			tenant_id=tenant,
			policy_number=data["policy_number"],
			msisdn=data["msisdn"],
			incident_description=data.get("incident_description", ""),
			claimed_amount=Decimal(str(data["claimed_amount"])),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@mic_bp.get("/claims")
def list_claims():
	tenant = request.args.get("tenant_id", "default")
	msisdn = request.args.get("msisdn")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_claims(tenant, msisdn, status)))


@mic_bp.post("/claims/<claim_id>/payout")
def process_payout(claim_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.process_mobile_payout(
			tenant_id=tenant,
			claim_id=claim_id,
			msisdn=data["msisdn"],
			amount=Decimal(str(data["amount"])),
			operator=data["operator"],
			mobile_money_reference=data["mobile_money_reference"],
		))
		return jsonify(rec), 201
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@mic_bp.post("/enrolments/<enrolment_id>/renew")
def renew_enrolment(enrolment_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.renew_enrolment(tenant, enrolment_id, data.get("payment_method")))
		return jsonify(rec), 201
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@mic_bp.get("/summary")
def platform_summary():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.platform_summary(tenant)))


@mic_bp.get("/audit")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant)))
