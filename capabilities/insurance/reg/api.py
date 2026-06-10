"""Flask Blueprint REST API for Insurance Regulatory Reporting (ins_reg)."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from flask import Blueprint, jsonify, request

from .service import InsuranceRegulatoryReportingService

_log = logging.getLogger(__name__)

reg_bp = Blueprint("ins_reg", __name__, url_prefix="/api/insurance/reg")
_svc = InsuranceRegulatoryReportingService()


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@reg_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@reg_bp.get("/describe")
def describe():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.describe(tenant)))


@reg_bp.get("/returns")
def list_returns():
	tenant = request.args.get("tenant_id", "default")
	regulator = request.args.get("regulator")
	return_type = request.args.get("return_type")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_returns(tenant, regulator, return_type, status)))


@reg_bp.post("/returns")
def create_return():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.create_return(
			tenant_id=tenant,
			return_type=data["return_type"],
			regulator=data["regulator"],
			period_start=data["period_start"],
			period_end=data["period_end"],
			prepared_by=data.get("prepared_by", ""),
			data=data.get("data", {}),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@reg_bp.get("/returns/<return_id>")
def get_return(return_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_return(tenant, return_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@reg_bp.put("/returns/<return_id>")
def update_return(return_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_return(tenant, return_id, data)))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@reg_bp.delete("/returns/<return_id>")
def delete_return(return_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_return(tenant, return_id)))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@reg_bp.post("/returns/<return_id>/review")
def review_return(return_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.review_return(tenant, return_id, data.get("reviewed_by", ""), data.get("notes", ""))))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@reg_bp.post("/returns/<return_id>/submit")
def submit_return(return_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.submit_return(tenant, return_id, data.get("submitted_by", ""), data.get("submission_channel", "portal"))))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@reg_bp.post("/returns/<return_id>/accept")
def accept_return(return_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.record_acceptance(tenant, return_id, data.get("regulator_reference", ""))))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@reg_bp.post("/solvency")
def prepare_solvency():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.prepare_solvency_report(
			tenant_id=tenant,
			valuation_date=data["valuation_date"],
			total_assets=Decimal(str(data["total_assets"])),
			total_liabilities=Decimal(str(data["total_liabilities"])),
			eligible_own_funds=Decimal(str(data["eligible_own_funds"])),
			scr=Decimal(str(data["scr"])),
			mcr=Decimal(str(data["mcr"])),
			prepared_by=data.get("prepared_by", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@reg_bp.get("/solvency")
def list_solvency():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.list_solvency_reports(tenant)))


@reg_bp.post("/statistical")
def compile_statistical():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.compile_statistical_return(
			tenant_id=tenant,
			period=data["period"],
			policies_in_force=int(data["policies_in_force"]),
			gross_premium=Decimal(str(data["gross_premium"])),
			net_premium=Decimal(str(data["net_premium"])),
			gross_claims=Decimal(str(data["gross_claims"])),
			net_claims=Decimal(str(data["net_claims"])),
			prepared_by=data.get("prepared_by", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@reg_bp.post("/market-conduct")
def file_market_conduct():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.file_market_conduct(
			tenant_id=tenant,
			filing_type=data["filing_type"],
			subject=data["subject"],
			description=data.get("description", ""),
			submitted_by=data.get("submitted_by", ""),
			attachments=data.get("attachments", []),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@reg_bp.get("/market-conduct")
def list_market_conduct():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.list_market_conduct_filings(tenant)))


@reg_bp.get("/calendar")
def compliance_calendar():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.list_compliance_calendar(tenant)))


@reg_bp.get("/calendar/upcoming")
def upcoming_deadlines():
	tenant = request.args.get("tenant_id", "default")
	days = int(request.args.get("days", 30))
	return jsonify(_run(_svc.list_upcoming_deadlines(tenant, days)))


@reg_bp.post("/calendar")
def add_deadline():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.add_compliance_deadline(
			tenant_id=tenant,
			return_type=data["return_type"],
			regulator=data["regulator"],
			due_date=data["due_date"],
			frequency=data.get("frequency", "annual"),
			responsible_party=data.get("responsible_party", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@reg_bp.get("/summary")
def regulatory_summary():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.regulatory_summary(tenant)))


@reg_bp.get("/audit")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant)))
