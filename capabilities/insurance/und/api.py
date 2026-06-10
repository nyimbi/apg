"""Flask Blueprint REST API for Underwriting Engine (ins_und)."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from flask import Blueprint, jsonify, request

from .service import UnderwritingEngineService

_log = logging.getLogger(__name__)

und_bp = Blueprint("ins_und", __name__, url_prefix="/api/insurance/und")
_svc = UnderwritingEngineService()


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@und_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@und_bp.get("/describe")
def describe():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.describe(tenant)))


@und_bp.get("/submissions")
def list_submissions():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_submissions(tenant, status)))


@und_bp.post("/submissions")
def submit_risk():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.submit_risk(
			tenant_id=tenant,
			proposer_name=data["proposer_name"],
			proposer_id=data["proposer_id"],
			product_code=data["product_code"],
			risk_class=data.get("risk_class", "standard"),
			sum_insured=Decimal(str(data["sum_insured"])),
			submitted_by=data.get("submitted_by", ""),
			currency=data.get("currency", "KES"),
			risk_attributes=data.get("risk_attributes", {}),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@und_bp.get("/submissions/<submission_id>")
def get_submission(submission_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_submission(tenant, submission_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@und_bp.post("/submissions/<submission_id>/assess")
def assess_risk(submission_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.assess_risk(tenant, submission_id, data.get("assessed_by")))
		return jsonify(rec), 201
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@und_bp.post("/submissions/<submission_id>/rate")
def rate_risk(submission_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		adj = {k: Decimal(str(v)) for k, v in data.get("adjustments", {}).items()}
		rec = _run(_svc.rate_risk(
			tenant_id=tenant,
			submission_id=submission_id,
			base_rate=Decimal(str(data["base_rate"])),
			adjustments=adj,
			rated_by=data.get("rated_by", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@und_bp.post("/capacity/check")
def check_capacity():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.check_capacity(
			tenant_id=tenant,
			product_code=data["product_code"],
			risk_class=data.get("risk_class", "standard"),
			requested_sum_insured=Decimal(str(data["requested_sum_insured"])),
			currency=data.get("currency", "KES"),
		))
		return jsonify(rec)
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@und_bp.get("/treaties")
def list_treaties():
	tenant = request.args.get("tenant_id", "default")
	active_only = request.args.get("active_only", "true").lower() == "true"
	return jsonify(_run(_svc.list_treaties(tenant, active_only)))


@und_bp.post("/treaties")
def create_treaty():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.create_treaty(
			tenant_id=tenant,
			treaty_name=data["treaty_name"],
			treaty_type=data["treaty_type"],
			reinsurer=data["reinsurer"],
			retention=Decimal(str(data["retention"])),
			cession_pct=float(data["cession_pct"]),
			treaty_limit=Decimal(str(data["treaty_limit"])),
			effective_date=data["effective_date"],
			expiry_date=data["expiry_date"],
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@und_bp.get("/rules")
def list_rules():
	tenant = request.args.get("tenant_id", "default")
	product_code = request.args.get("product_code")
	return jsonify(_run(_svc.list_rules(tenant, product_code)))


@und_bp.post("/rules")
def create_rule():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.create_rule(
			tenant_id=tenant,
			rule_name=data["rule_name"],
			product_code=data["product_code"],
			condition=data["condition"],
			action=data["action"],
			priority=data.get("priority", 100),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@und_bp.delete("/rules/<rule_id>")
def delete_rule(rule_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_rule(tenant, rule_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@und_bp.get("/summary")
def underwriting_summary():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.underwriting_summary(tenant)))


@und_bp.get("/audit")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant)))
