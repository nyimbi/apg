"""Flask Blueprint REST API for Policy Administration (ins_pol)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import PolicyAdministrationService

_log = logging.getLogger(__name__)

pol_bp = Blueprint("ins_pol", __name__, url_prefix="/api/insurance/pol")
_svc = PolicyAdministrationService()


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@pol_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@pol_bp.get("/describe")
def describe():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.describe(tenant)))


@pol_bp.get("/policies")
def list_policies():
	tenant = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	product_code = request.args.get("product_code")
	return jsonify(_run(_svc.list_policies(tenant, status, product_code)))


@pol_bp.post("/policies")
def create_policy():
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		from decimal import Decimal
		rec = _run(_svc.create_policy(
			tenant_id=tenant,
			policy_number=data["policy_number"],
			product_code=data["product_code"],
			insured_name=data["insured_name"],
			insured_id=data["insured_id"],
			sum_insured=Decimal(str(data["sum_insured"])),
			inception_date=data["inception_date"],
			expiry_date=data["expiry_date"],
			premium=Decimal(str(data["premium"])),
			underwriter_id=data["underwriter_id"],
			currency=data.get("currency", "KES"),
			agent_id=data.get("agent_id"),
			metadata=data.get("metadata", {}),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("create_policy: %s", exc)
		return jsonify({"error": str(exc)}), 500


@pol_bp.get("/policies/<policy_id>")
def get_policy(policy_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_policy(tenant, policy_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@pol_bp.put("/policies/<policy_id>")
def update_policy(policy_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		updates = {k: v for k, v in data.items() if k != "tenant_id"}
		return jsonify(_run(_svc.update_policy(tenant, policy_id, updates)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except (ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@pol_bp.delete("/policies/<policy_id>")
def delete_policy(policy_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	reason = data.get("reason", "voided")
	try:
		return jsonify(_run(_svc.delete_policy(tenant, policy_id, reason)))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@pol_bp.post("/policies/<policy_id>/endorse")
def endorse_policy(policy_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	from decimal import Decimal
	try:
		rec = _run(_svc.create_endorsement(
			tenant_id=tenant,
			policy_id=policy_id,
			endorsement_type=data["endorsement_type"],
			effective_date=data["effective_date"],
			description=data.get("description", ""),
			change_in_premium=Decimal(str(data.get("change_in_premium", 0))),
			change_in_sum_insured=Decimal(str(data.get("change_in_sum_insured", 0))),
			requested_by=data.get("requested_by", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@pol_bp.post("/policies/<policy_id>/renew")
def renew_policy(policy_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	from decimal import Decimal
	try:
		rec = _run(_svc.initiate_renewal(
			tenant_id=tenant,
			policy_id=policy_id,
			new_expiry_date=data["new_expiry_date"],
			new_premium=Decimal(str(data["new_premium"])),
			initiated_by=data.get("initiated_by", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@pol_bp.post("/policies/<policy_id>/cancel")
def cancel_policy(policy_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	from decimal import Decimal
	try:
		rec = _run(_svc.cancel_policy(
			tenant_id=tenant,
			policy_id=policy_id,
			cancellation_date=data["cancellation_date"],
			reason=data["reason"],
			cancellation_type=data.get("cancellation_type", "voluntary"),
			refund_premium=data.get("refund_premium", True),
			authorised_by=data.get("authorised_by", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@pol_bp.post("/policies/<policy_id>/reinstate")
def reinstate_policy(policy_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	from decimal import Decimal
	try:
		rec = _run(_svc.reinstate_policy(
			tenant_id=tenant,
			policy_id=policy_id,
			reinstatement_date=data["reinstatement_date"],
			outstanding_premium=Decimal(str(data.get("outstanding_premium", 0))),
			reason=data.get("reason", ""),
			authorised_by=data.get("authorised_by", ""),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@pol_bp.post("/policies/<policy_id>/documents")
def generate_document(policy_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.get("tenant_id", "default")
	try:
		rec = _run(_svc.generate_document(
			tenant_id=tenant,
			policy_id=policy_id,
			document_type=data["document_type"],
			generated_by=data.get("generated_by", "system"),
		))
		return jsonify(rec), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@pol_bp.get("/portfolio/summary")
def portfolio_summary():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.portfolio_summary(tenant)))


@pol_bp.get("/audit")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant)))
