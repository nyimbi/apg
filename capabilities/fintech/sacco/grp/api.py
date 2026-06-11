"""Flask Blueprint REST API for SACCO Group Lending.

© 2025 Datacraft — Author: Nyimbi Odero
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date
from decimal import Decimal
from typing import Any

from flask import Blueprint, jsonify, request

from .service import GroupLendingService

_log = logging.getLogger(__name__)

bp = Blueprint("sacco_grp", __name__, url_prefix="/api/fintech/sacco/grp")
_svc = GroupLendingService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _date(val: str | None) -> date | None:
	return date.fromisoformat(val) if val else None


def _dec(val: Any) -> Decimal:
	return Decimal(str(val))


# ── Health ────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check())), 200


# ── Groups ────────────────────────────────────────────────────────────────────

@bp.get("/groups")
def list_groups():
	result = _run(_svc.list_groups(
		tenant_id=_tenant(),
		group_type=request.args.get("group_type"),
		active_only=request.args.get("active_only", "true").lower() == "true",
	))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/groups")
def register_group():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.register_group(
			tenant_id=_tenant(),
			name=body["name"],
			group_type=body["group_type"],
			registration_number=body.get("registration_number"),
			meeting_day=body.get("meeting_day"),
			meeting_frequency=body.get("meeting_frequency", "MONTHLY"),
			chairperson_member_id=body.get("chairperson_member_id"),
			secretary_member_id=body.get("secretary_member_id"),
			treasurer_member_id=body.get("treasurer_member_id"),
			metadata=body.get("metadata", {}),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/groups/<group_id>")
def get_group(group_id: str):
	try:
		return jsonify(_run(_svc.get_group(_tenant(), group_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/groups/<group_id>")
def update_group(group_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.update_group(_tenant(), group_id, body))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


# ── Members ───────────────────────────────────────────────────────────────────

@bp.post("/groups/<group_id>/members")
def add_group_member(group_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.add_group_member(
			tenant_id=_tenant(),
			group_id=group_id,
			member_id=body["member_id"],
			role=body.get("role", "MEMBER"),
			joining_date=_date(body.get("joining_date")),
			initial_contribution=_dec(body.get("initial_contribution", 0)),
		))
		return jsonify(result), 201
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/groups/<group_id>/members/<member_id>")
def remove_group_member(group_id: str, member_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.remove_group_member(
			tenant_id=_tenant(),
			group_id=group_id,
			member_id=member_id,
			exit_date=_date(body.get("exit_date")),
			reason=body.get("reason", ""),
			payout_amount=_dec(body.get("payout_amount", 0)),
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


# ── Contributions ─────────────────────────────────────────────────────────────

@bp.post("/groups/<group_id>/contributions")
def record_group_contribution(group_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.record_group_contribution(
			tenant_id=_tenant(),
			group_id=group_id,
			contributions=body["contributions"],
			meeting_date=_date(body.get("meeting_date")),
			contribution_type=body.get("contribution_type", "MONTHLY"),
			recorded_by=body.get("recorded_by"),
			notes=body.get("notes"),
		))
		return jsonify(result), 201
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/groups/<group_id>/savings")
def get_group_savings(group_id: str):
	try:
		return jsonify(_run(_svc.get_group_savings(_tenant(), group_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/groups/<group_id>/contributions")
def get_contribution_history(group_id: str):
	try:
		months = int(request.args.get("months", 12))
		return jsonify(_run(_svc.get_contribution_history(_tenant(), group_id, months))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/groups/<group_id>/compliance")
def get_contribution_compliance(group_id: str):
	try:
		return jsonify(_run(_svc.get_contribution_compliance(_tenant(), group_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Group loans ───────────────────────────────────────────────────────────────

@bp.get("/loans")
def list_group_loans():
	result = _run(_svc.list_group_loans(
		tenant_id=_tenant(),
		group_id=request.args.get("group_id"),
		status=request.args.get("status"),
	))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/loans")
def apply_group_loan():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.apply_group_loan(
			tenant_id=_tenant(),
			group_id=body["group_id"],
			requested_amount=_dec(body["requested_amount"]),
			purpose=body["purpose"],
			tenure_months=int(body["tenure_months"]),
			applied_by=body["applied_by"],
		))
		return jsonify(result), 201
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/loans/<loan_id>")
def get_group_loan(loan_id: str):
	try:
		return jsonify(_run(_svc.get_group_loan(_tenant(), loan_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/loans/<loan_id>/approve")
def approve_group_loan(loan_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.approve_group_loan(
			tenant_id=_tenant(),
			loan_application_id=loan_id,
			approved_amount=_dec(body["approved_amount"]),
			approved_by=body["approved_by"],
			conditions=body.get("conditions"),
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/loans/<loan_id>/disburse")
def disburse_group_loan(loan_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.disburse_group_loan(
			tenant_id=_tenant(),
			loan_id=loan_id,
			disbursement_instructions=body["disbursement_instructions"],
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/loans/<loan_id>/repayments")
def record_group_repayment(loan_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.record_group_repayment(
			tenant_id=_tenant(),
			loan_id=loan_id,
			total_amount=_dec(body["total_amount"]),
			payment_date=_date(body.get("payment_date")),
			payment_ref=body["payment_ref"],
			member_contributions=body["member_contributions"],
			notes=body.get("notes"),
		))
		return jsonify(result), 201
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/loans/<loan_id>/arrears")
def calculate_group_loan_arrears(loan_id: str):
	try:
		as_of = _date(request.args.get("as_of_date"))
		return jsonify(_run(_svc.calculate_group_loan_arrears(_tenant(), loan_id, as_of))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/loans/<loan_id>/defaulting-members")
def get_defaulting_members(loan_id: str):
	try:
		return jsonify(_run(_svc.get_defaulting_members(_tenant(), loan_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/loans/<loan_id>/joint-liability")
def trigger_joint_liability(loan_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.trigger_joint_liability(
			tenant_id=_tenant(),
			loan_id=loan_id,
			defaulting_member_id=body["defaulting_member_id"],
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


# ── Merry-go-round ────────────────────────────────────────────────────────────

@bp.get("/groups/<group_id>/mgr/schedule")
def get_merry_go_round_schedule(group_id: str):
	try:
		return jsonify(_run(_svc.get_merry_go_round_schedule(_tenant(), group_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/groups/<group_id>/mgr/order")
def set_merry_go_round_order(group_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.set_merry_go_round_order(
			tenant_id=_tenant(),
			group_id=group_id,
			member_order=body["member_order"],
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/groups/<group_id>/mgr/process")
def process_merry_go_round(group_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.process_merry_go_round(
			tenant_id=_tenant(),
			group_id=group_id,
			round_date=_date(body.get("round_date")),
			beneficiary_member_id=body["beneficiary_member_id"],
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


# ── Reporting ─────────────────────────────────────────────────────────────────

@bp.get("/groups/<group_id>/performance")
def get_group_performance_score(group_id: str):
	try:
		return jsonify(_run(_svc.get_group_performance_score(_tenant(), group_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/groups/<group_id>/statement")
def get_group_statement(group_id: str):
	try:
		from_date = date.fromisoformat(request.args["from_date"])
		to_date = date.fromisoformat(request.args["to_date"])
		return jsonify(_run(_svc.get_group_statement(_tenant(), group_id, from_date, to_date))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 400
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400
