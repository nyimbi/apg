"""Flask Blueprint REST API for SACCO Check-off Management."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import CheckOffService

_log = logging.getLogger(__name__)

bp = Blueprint("sacco_ckf", __name__, url_prefix="/api/fintech/sacco/ckf")
_svc = CheckOffService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _int(name: str, default: int) -> int:
	try:
		return int(request.args.get(name, default))
	except (TypeError, ValueError):
		return default


# ── Health ────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check())), 200


# ── Employers ─────────────────────────────────────────────────────────────────

@bp.get("/employers")
def list_employers():
	active_only = request.args.get("active_only", "true").lower() == "true"
	result = _run(_svc.list_employers(tenant_id=_tenant(), active_only=active_only))
	return jsonify({"items": [e.model_dump() for e in result], "total": len(result)}), 200


@bp.get("/employers/<employer_id>")
def get_employer(employer_id: str):
	try:
		return jsonify(_run(_svc.get_employer(_tenant(), employer_id)).model_dump()), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/employers")
def register_employer():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.register_employer(
			tenant_id=_tenant(),
			name=body["name"],
			registration_number=body["registration_number"],
			payroll_contact=body["payroll_contact"],
			remittance_account=body["remittance_account"],
			check_off_agreement_date=body["check_off_agreement_date"],
			deduction_frequency=body.get("deduction_frequency", "monthly"),
			email=body.get("email"),
			phone=body.get("phone"),
			address=body.get("address"),
			notes=body.get("notes"),
		))
		return jsonify(result.model_dump()), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.patch("/employers/<employer_id>")
def update_employer(employer_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.update_employer(_tenant(), employer_id, body))
		return jsonify(result.model_dump()), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/employers/<employer_id>/deactivate")
def deactivate_employer(employer_id: str):
	body = request.get_json(force=True) or {}
	reason = body.get("reason", "")
	if not reason:
		return jsonify({"error": "reason_required"}), 400
	try:
		result = _run(_svc.deactivate_employer(_tenant(), employer_id, reason))
		return jsonify(result.model_dump()), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Member-Employer Links ─────────────────────────────────────────────────────

@bp.post("/links")
def add_link():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.add_member_employer_link(
			tenant_id=_tenant(),
			member_id=body["member_id"],
			employer_id=body["employer_id"],
			employee_number=body["employee_number"],
			basic_salary=body["basic_salary"],
			effective_date=body["effective_date"],
			member_name=body.get("member_name", ""),
		))
		return jsonify(result.model_dump()), 201
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/links")
def remove_link():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.remove_member_employer_link(
			tenant_id=_tenant(),
			member_id=body["member_id"],
			employer_id=body["employer_id"],
			effective_date=body["effective_date"],
			reason=body["reason"],
		))
		return jsonify(result.model_dump()), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/members/<member_id>/deductions")
def member_deductions(member_id: str):
	try:
		result = _run(_svc.get_member_deductions(_tenant(), member_id))
		return jsonify(result.model_dump()), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/members/<member_id>/history")
def member_history(member_id: str):
	months = _int("months", 12)
	result = _run(_svc.get_member_check_off_history(_tenant(), member_id, months=months))
	return jsonify(result.model_dump()), 200


# ── Schedule ──────────────────────────────────────────────────────────────────

@bp.post("/employers/<employer_id>/schedule")
def generate_schedule(employer_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.generate_check_off_schedule(
			tenant_id=_tenant(),
			employer_id=employer_id,
			payroll_month=int(body["payroll_month"]),
			payroll_year=int(body["payroll_year"]),
		))
		return jsonify(result.model_dump()), 200
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Upload ────────────────────────────────────────────────────────────────────

@bp.post("/employers/<employer_id>/upload")
def upload_file(employer_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.upload_check_off_file(
			tenant_id=_tenant(),
			employer_id=employer_id,
			payroll_month=int(body["payroll_month"]),
			payroll_year=int(body["payroll_year"]),
			deductions=body.get("deductions", []),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Reconciliation ────────────────────────────────────────────────────────────

@bp.post("/employers/<employer_id>/reconcile")
def reconcile(employer_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.reconcile_check_off(
			tenant_id=_tenant(),
			employer_id=employer_id,
			payroll_month=int(body["payroll_month"]),
			payroll_year=int(body["payroll_year"]),
		))
		return jsonify(result.model_dump()), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Post Receipts ─────────────────────────────────────────────────────────────

@bp.post("/employers/<employer_id>/post")
def post_receipts(employer_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.post_check_off_receipts(
			tenant_id=_tenant(),
			employer_id=employer_id,
			payroll_month=int(body["payroll_month"]),
			payroll_year=int(body["payroll_year"]),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Status & Queries ──────────────────────────────────────────────────────────

@bp.get("/employers/<employer_id>/status")
def check_off_status(employer_id: str):
	month = _int("month", 1)
	year = _int("year", 2026)
	try:
		result = _run(_svc.get_check_off_status(_tenant(), employer_id, month, year))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/outstanding")
def outstanding_remittances():
	result = _run(_svc.get_outstanding_remittances(_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/employers/<employer_id>/remind")
def send_reminder(employer_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.send_remittance_reminder(
			tenant_id=_tenant(),
			employer_id=employer_id,
			payroll_month=int(body["payroll_month"]),
			payroll_year=body.get("payroll_year"),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/employers/<employer_id>/statement")
def employer_statement(employer_id: str):
	from_month = _int("from_month", 1)
	to_month = _int("to_month", 12)
	from_year = _int("from_year", 0) or None
	to_year = _int("to_year", 0) or None
	try:
		result = _run(_svc.generate_employer_statement(
			tenant_id=_tenant(),
			employer_id=employer_id,
			from_month=from_month,
			to_month=to_month,
			from_year=from_year,
			to_year=to_year,
		))
		return jsonify(result.model_dump()), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/employers/<employer_id>/default")
def flag_default(employer_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.flag_employer_default(
			tenant_id=_tenant(),
			employer_id=employer_id,
			defaulted_month=int(body["defaulted_month"]),
			defaulted_year=body.get("defaulted_year"),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/metrics")
def metrics():
	month = _int("month", 0) or None
	year = _int("year", 0) or None
	result = _run(_svc.get_check_off_metrics(_tenant(), month, year))
	return jsonify(result.model_dump()), 200


@bp.post("/batch-schedule")
def batch_schedule():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.batch_process_all_employers(
			tenant_id=_tenant(),
			payroll_month=int(body["payroll_month"]),
			payroll_year=int(body["payroll_year"]),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
