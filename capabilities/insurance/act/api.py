"""Flask Blueprint REST API for Actuarial Tools (ins_act)."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from flask import Blueprint, jsonify, request

from .service import ActuarialToolsService

_log = logging.getLogger(__name__)

act_bp = Blueprint("ins_act", __name__, url_prefix="/api/insurance/act")
_svc = ActuarialToolsService()


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@act_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@act_bp.get("/describe")
def describe():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.describe(tenant)))


@act_bp.get("/mortality-tables")
def list_tables():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.list_mortality_tables(tenant)))


@act_bp.post("/mortality-tables")
def create_table():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.create_mortality_table(
			tenant_id=tenant,
			table_name=data["table_name"],
			table_type=data.get("table_type", "population"),
			base_year=int(data["base_year"]),
			ages=data["ages"],
			qx_values=data["qx_values"],
			lx_values=data["lx_values"],
			source=data.get("source", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@act_bp.get("/mortality-tables/<table_id>")
def get_table(table_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_mortality_table(tenant, table_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@act_bp.delete("/mortality-tables/<table_id>")
def delete_table(table_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_mortality_table(tenant, table_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@act_bp.post("/loss-ratio")
def calculate_loss_ratio():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.calculate_loss_ratio(
			tenant_id=tenant,
			product_code=data["product_code"],
			period_start=data["period_start"],
			period_end=data["period_end"],
			earned_premium=Decimal(str(data["earned_premium"])),
			incurred_losses=Decimal(str(data["incurred_losses"])),
			expenses=Decimal(str(data.get("expenses", 0))),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@act_bp.get("/loss-ratios")
def list_loss_ratios():
	tenant = request.args.get("tenant_id", "default")
	product_code = request.args.get("product_code")
	return jsonify(_run(_svc.list_loss_ratios(tenant, product_code)))


@act_bp.post("/reserves")
def calculate_reserve():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.calculate_reserve(
			tenant_id=tenant,
			product_code=data["product_code"],
			valuation_date=data["valuation_date"],
			method=data.get("method", "chain_ladder"),
			gross_claims_paid=Decimal(str(data["gross_claims_paid"])),
			gross_claims_outstanding=Decimal(str(data["gross_claims_outstanding"])),
			reinsurance_recoverable=Decimal(str(data.get("reinsurance_recoverable", 0))),
			assumptions=data.get("assumptions", {}),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@act_bp.get("/reserves")
def list_reserves():
	tenant = request.args.get("tenant_id", "default")
	product_code = request.args.get("product_code")
	return jsonify(_run(_svc.list_reserves(tenant, product_code)))


@act_bp.post("/ibnr")
def estimate_ibnr():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.estimate_ibnr(
			tenant_id=tenant,
			product_code=data["product_code"],
			valuation_date=data["valuation_date"],
			development_method=data.get("development_method", "chain_ladder"),
			triangle_data=data.get("triangle_data", []),
			confidence_level=float(data.get("confidence_level", 0.75)),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@act_bp.get("/ibnr")
def list_ibnr():
	tenant = request.args.get("tenant_id", "default")
	product_code = request.args.get("product_code")
	return jsonify(_run(_svc.list_ibnr_estimates(tenant, product_code)))


@act_bp.post("/pricing-models")
def create_pricing_model():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.create_pricing_model(
			tenant_id=tenant,
			model_name=data["model_name"],
			product_code=data["product_code"],
			risk_factors=data.get("risk_factors", []),
			base_rate=Decimal(str(data["base_rate"])),
			parameters=data.get("parameters", {}),
			effective_date=data.get("effective_date"),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@act_bp.get("/pricing-models")
def list_pricing_models():
	tenant = request.args.get("tenant_id", "default")
	product_code = request.args.get("product_code")
	return jsonify(_run(_svc.list_pricing_models(tenant, product_code)))


@act_bp.post("/pricing-models/<model_id>/apply")
def apply_pricing_model(model_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.apply_pricing_model(tenant, model_id, data.get("risk_data", {}))))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@act_bp.post("/experience-analysis")
def run_experience_analysis():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.run_experience_analysis(
			tenant_id=tenant,
			product_code=data["product_code"],
			analysis_period_years=int(data.get("analysis_period_years", 3)),
			actual_claims=int(data["actual_claims"]),
			expected_claims=int(data["expected_claims"]),
			actual_loss_amount=Decimal(str(data["actual_loss_amount"])),
			expected_loss_amount=Decimal(str(data["expected_loss_amount"])),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@act_bp.get("/experience-analyses")
def list_experience_analyses():
	tenant = request.args.get("tenant_id", "default")
	product_code = request.args.get("product_code")
	return jsonify(_run(_svc.list_experience_analyses(tenant, product_code)))


@act_bp.get("/summary")
def actuarial_summary():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.actuarial_summary(tenant)))


@act_bp.get("/audit")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant)))
