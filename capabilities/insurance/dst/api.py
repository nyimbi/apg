"""Flask Blueprint REST API for Distribution & Agency Management (ins_dst)."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from flask import Blueprint, jsonify, request

from .service import DistributionAgencyService

_log = logging.getLogger(__name__)

dst_bp = Blueprint("ins_dst", __name__, url_prefix="/api/insurance/dst")
_svc = DistributionAgencyService()


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@dst_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@dst_bp.get("/describe")
def describe():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.describe(tenant)))


@dst_bp.get("/agents")
def list_agents():
	tenant = request.args.get("tenant_id", "default")
	agent_type = request.args.get("agent_type")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_agents(tenant, agent_type, status)))


@dst_bp.post("/agents")
def register_agent():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.register_agent(
			tenant_id=tenant,
			agent_code=data["agent_code"],
			agent_name=data["agent_name"],
			agent_type=data["agent_type"],
			id_number=data["id_number"],
			ira_licence_number=data["ira_licence_number"],
			phone=data["phone"],
			email=data["email"],
			supervisor_id=data.get("supervisor_id"),
			branch_id=data.get("branch_id"),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@dst_bp.get("/agents/<agent_id>")
def get_agent(agent_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_agent(tenant, agent_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@dst_bp.put("/agents/<agent_id>")
def update_agent(agent_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_agent(tenant, agent_id, data)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@dst_bp.delete("/agents/<agent_id>")
def delete_agent(agent_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_agent(tenant, agent_id, data.get("reason", "deregistered"))))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@dst_bp.post("/agents/<agent_id>/suspend")
def suspend_agent(agent_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.suspend_agent(tenant, agent_id, data.get("reason", ""), data.get("suspended_by", "")))
		return jsonify(rec)
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@dst_bp.post("/commissions")
def compute_commission():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rate = Decimal(str(data["commission_rate"])) if "commission_rate" in data else None
		rec = _run(_svc.compute_commission(
			tenant_id=tenant,
			agent_id=data["agent_id"],
			policy_id=data["policy_id"],
			policy_number=data["policy_number"],
			product_code=data["product_code"],
			premium_amount=Decimal(str(data["premium_amount"])),
			commission_rate=rate,
			period=data.get("period", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@dst_bp.get("/commissions")
def list_commissions():
	tenant = request.args.get("tenant_id", "default")
	agent_id = request.args.get("agent_id")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_commissions(tenant, agent_id, status)))


@dst_bp.get("/commissions/<commission_id>")
def get_commission(commission_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_commission(tenant, commission_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@dst_bp.post("/commissions/<commission_id>/approve")
def approve_commission(commission_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.approve_commission(tenant, commission_id, data.get("approved_by", ""))))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@dst_bp.post("/commissions/<commission_id>/pay")
def pay_commission(commission_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.pay_commission(tenant, commission_id, data.get("payment_reference", ""))))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@dst_bp.post("/compliance")
def record_compliance():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.record_compliance(
			tenant_id=tenant,
			agent_id=data["agent_id"],
			compliance_type=data["compliance_type"],
			status=data.get("status", "compliant"),
			expiry_date=data.get("expiry_date"),
			notes=data.get("notes", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@dst_bp.get("/compliance")
def list_compliance():
	tenant = request.args.get("tenant_id", "default")
	agent_id = request.args.get("agent_id")
	return jsonify(_run(_svc.list_compliance_records(tenant, agent_id)))


@dst_bp.post("/bancassurance")
def register_bancassurance():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.register_bancassurance_partner(
			tenant_id=tenant,
			partner_name=data["partner_name"],
			partner_type=data.get("partner_type", "bank"),
			bank_code=data["bank_code"],
			products=data.get("products", []),
			commission_rate=Decimal(str(data["commission_rate"])),
			effective_date=data["effective_date"],
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@dst_bp.get("/bancassurance")
def list_bancassurance():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.list_bancassurance_partners(tenant)))


@dst_bp.post("/performance/<agent_id>")
def generate_performance_report(agent_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		target = Decimal(str(data["target_premium"])) if "target_premium" in data else None
		rec = _run(_svc.generate_performance_report(
			tenant_id=tenant,
			agent_id=agent_id,
			period_start=data["period_start"],
			period_end=data["period_end"],
			target_premium=target,
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@dst_bp.get("/summary")
def agency_summary():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.agency_summary(tenant)))


@dst_bp.get("/audit")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant)))
