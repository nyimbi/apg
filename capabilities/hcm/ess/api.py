"""Flask Blueprint REST API for Employee Self-Service."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import ESSService

_log = logging.getLogger(__name__)

bp = Blueprint("hcm_ess", __name__, url_prefix="/api/hcm/ess")
_svc = ESSService()


def _run(coro: Any) -> Any:
	"""Run an async coroutine synchronously (Flask compat helper)."""
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@bp.get("/describe")
def describe():
	return jsonify(_run(_svc.describe()))


# ── Leave Requests ────────────────────────────────────────────────────────────

@bp.get("/leave-requests")
def list_leave_requests():
	tenant_id = request.args.get("tenant_id", "default")
	employee_id = request.args.get("employee_id")
	status = request.args.get("status")
	leave_type = request.args.get("leave_type")
	try:
		items = _run(_svc.list_leave_requests(tenant_id, employee_id=employee_id, status=status, leave_type=leave_type))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		_log.error("list_leave_requests: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.get("/leave-requests/<request_id>")
def get_leave_request(request_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_leave_request(tenant_id, request_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("get_leave_request: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.post("/leave-requests")
def create_leave_request():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_leave_request(
			tenant_id=data.get("tenant_id", "default"),
			employee_id=data["employee_id"],
			leave_type=data["leave_type"],
			start_date=data["start_date"],
			end_date=data["end_date"],
			reason=data.get("reason"),
			handover_to=data.get("handover_to"),
			attachments=data.get("attachments"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("create_leave_request: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.put("/leave-requests/<request_id>/approve")
def approve_leave_request(request_id: str):
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.approve_leave_request(
			data.get("tenant_id", "default"), request_id, data["approved_by"]
		))
		return jsonify(record)
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("approve_leave_request: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.put("/leave-requests/<request_id>/reject")
def reject_leave_request(request_id: str):
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.reject_leave_request(
			data.get("tenant_id", "default"), request_id,
			data["rejected_by"], data["rejection_reason"]
		))
		return jsonify(record)
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("reject_leave_request: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.put("/leave-requests/<request_id>/cancel")
def cancel_leave_request(request_id: str):
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.cancel_leave_request(data.get("tenant_id", "default"), request_id))
		return jsonify(record)
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/leave-requests/<request_id>")
def delete_leave_request(request_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_leave_request(tenant_id, request_id))
		return jsonify({"deleted": True}), 200
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Payslips ──────────────────────────────────────────────────────────────────

@bp.get("/payslips")
def list_payslips():
	tenant_id = request.args.get("tenant_id", "default")
	employee_id = request.args.get("employee_id", "")
	year = request.args.get("year")
	try:
		items = _run(_svc.list_payslips(tenant_id, employee_id, year=int(year) if year else None))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/payslips/<payslip_id>")
def get_payslip(payslip_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_payslip(tenant_id, payslip_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/payslips")
def generate_payslip():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.generate_payslip(
			tenant_id=data.get("tenant_id", "default"),
			employee_id=data["employee_id"],
			period_month=int(data["period_month"]),
			period_year=int(data["period_year"]),
			gross_pay=float(data["gross_pay"]),
			earnings_breakdown=data.get("earnings_breakdown"),
			deductions_breakdown=data.get("deductions_breakdown"),
			currency=data.get("currency", "KES"),
			pay_date=data.get("pay_date"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Expense Claims ────────────────────────────────────────────────────────────

@bp.get("/expense-claims")
def list_expense_claims():
	tenant_id = request.args.get("tenant_id", "default")
	employee_id = request.args.get("employee_id")
	status = request.args.get("status")
	try:
		items = _run(_svc.list_expense_claims(tenant_id, employee_id=employee_id, status=status))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/expense-claims/<claim_id>")
def get_expense_claim(claim_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_expense_claim(tenant_id, claim_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/expense-claims")
def create_expense_claim():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_expense_claim(
			tenant_id=data.get("tenant_id", "default"),
			employee_id=data["employee_id"],
			category=data["category"],
			amount=float(data["amount"]),
			expense_date=data["expense_date"],
			description=data["description"],
			currency=data.get("currency", "KES"),
			receipts=data.get("receipts"),
			project_code=data.get("project_code"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/expense-claims/<claim_id>")
def update_expense_claim(claim_id: str):
	data = request.get_json(force=True) or {}
	tenant_id = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_expense_claim(tenant_id, claim_id, **data)))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/expense-claims/<claim_id>")
def delete_expense_claim(claim_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_expense_claim(tenant_id, claim_id))
		return jsonify({"deleted": True})
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Benefits ──────────────────────────────────────────────────────────────────

@bp.get("/benefit-enrolments")
def list_benefit_enrolments():
	tenant_id = request.args.get("tenant_id", "default")
	employee_id = request.args.get("employee_id")
	try:
		items = _run(_svc.list_benefit_enrolments(tenant_id, employee_id=employee_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/benefit-enrolments")
def enrol_benefit():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.enrol_benefit(
			tenant_id=data.get("tenant_id", "default"),
			employee_id=data["employee_id"],
			benefit_plan_id=data["benefit_plan_id"],
			benefit_type=data["benefit_type"],
			effective_date=data["effective_date"],
			coverage_tier=data.get("coverage_tier", "individual"),
			dependants=data.get("dependants"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Training ──────────────────────────────────────────────────────────────────

@bp.get("/training-registrations")
def list_training_registrations():
	tenant_id = request.args.get("tenant_id", "default")
	employee_id = request.args.get("employee_id")
	status = request.args.get("status")
	try:
		items = _run(_svc.list_training_registrations(tenant_id, employee_id=employee_id, status=status))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/training-registrations")
def register_training():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.register_training(
			tenant_id=data.get("tenant_id", "default"),
			employee_id=data["employee_id"],
			course_id=data["course_id"],
			course_name=data["course_name"],
			training_type=data["training_type"],
			start_date=data["start_date"],
			end_date=data["end_date"],
			provider=data.get("provider"),
			cost=float(data.get("cost", 0.0)),
			currency=data.get("currency", "KES"),
			justification=data.get("justification"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Dashboard ─────────────────────────────────────────────────────────────────

@bp.get("/dashboard")
def dashboard():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.dashboard_summary(tenant_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/audit-events")
def audit_events():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		events = _run(_svc.get_audit_events(tenant_id))
		return jsonify({"items": events, "total": len(events)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400
