"""Flask Blueprint REST API for Premium & Billing (ins_prm)."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from flask import Blueprint, jsonify, request

from .service import PremiumBillingService

_log = logging.getLogger(__name__)

prm_bp = Blueprint("ins_prm", __name__, url_prefix="/api/insurance/prm")
_svc = PremiumBillingService()


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@prm_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@prm_bp.get("/describe")
def describe():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.describe(tenant)))


@prm_bp.get("/schedules")
def list_schedules():
	tenant = request.args.get("tenant_id", "default")
	policy_id = request.args.get("policy_id")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_schedules(tenant, policy_id, status)))


@prm_bp.post("/schedules")
def create_schedule():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.create_schedule(
			tenant_id=tenant,
			policy_id=data["policy_id"],
			policy_number=data["policy_number"],
			total_premium=Decimal(str(data["total_premium"])),
			frequency=data.get("frequency", "annual"),
			inception_date=data["inception_date"],
			expiry_date=data["expiry_date"],
			currency=data.get("currency", "KES"),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@prm_bp.get("/schedules/<schedule_id>")
def get_schedule(schedule_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_schedule(tenant, schedule_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@prm_bp.delete("/schedules/<schedule_id>")
def delete_schedule(schedule_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_schedule(tenant, schedule_id)))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@prm_bp.get("/instalments")
def list_instalments():
	tenant = request.args.get("tenant_id", "default")
	schedule_id = request.args.get("schedule_id")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_instalments(tenant, schedule_id, status)))


@prm_bp.get("/instalments/overdue")
def overdue_instalments():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.list_overdue_instalments(tenant)))


@prm_bp.post("/instalments/<instalment_id>/collect")
def collect_payment(instalment_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.collect_payment(
			tenant_id=tenant,
			instalment_id=instalment_id,
			payment_method=data["payment_method"],
			payment_reference=data["payment_reference"],
			amount=Decimal(str(data["amount"])),
			collected_by=data.get("collected_by", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@prm_bp.post("/refunds")
def process_refund():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.process_refund(
			tenant_id=tenant,
			policy_id=data["policy_id"],
			refund_amount=Decimal(str(data["refund_amount"])),
			reason=data["reason"],
			payee_account=data["payee_account"],
			authorised_by=data.get("authorised_by", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@prm_bp.post("/reconcile")
def reconcile():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.reconcile_period(
			tenant_id=tenant,
			period_start=data["period_start"],
			period_end=data["period_end"],
			reconciled_by=data.get("reconciled_by", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@prm_bp.post("/calculate")
def calculate_premium():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		loadings = {k: Decimal(str(v)) for k, v in data.get("loadings", {}).items()}
		discounts = {k: Decimal(str(v)) for k, v in data.get("discounts", {}).items()}
		result = _run(_svc.calculate_premium(
			tenant_id=tenant,
			product_code=data["product_code"],
			sum_insured=Decimal(str(data["sum_insured"])),
			base_rate=Decimal(str(data["base_rate"])),
			loadings=loadings,
			discounts=discounts,
		))
		return jsonify(result)
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@prm_bp.get("/summary")
def billing_summary():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.billing_summary(tenant)))


@prm_bp.get("/audit")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant)))
