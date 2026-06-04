"""Flask Blueprint views for Real Estate Accounting (acc)."""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, request, jsonify, g

from .service import AccService
from .models import (
	AccountCreate, AccountUpdate,
	JournalEntryCreate,
	ServiceChargeCreate,
	CamReconciliationCreate,
	Ifrs16ScheduleCreate,
	RevenueScheduleCreate,
	AccountingPeriodCreate,
	TenantStatementCreate,
)

bp = Blueprint("acc_views", __name__, url_prefix="/realestate/acc")
_svc = AccService()


def _run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def has_access(permission: str):
	"""Stub permission decorator — replace with real auth middleware."""
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args, **kwargs):
			# Real implementation checks g.user permissions
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _ok(data: Any, status: int = 200):
	return jsonify({"status": "ok", "data": data}), status


def _err(msg: str, status: int = 400):
	return jsonify({"status": "error", "message": msg}), status


# ── Dashboard ─────────────────────────────────────────────────────────────────

@bp.get("/dashboard")
@has_access("realestate_acc:view")
def dashboard():
	summary = _run(_svc.get_financial_summary(_tenant()))
	return _ok(summary)


# ── Accounts ──────────────────────────────────────────────────────────────────

@bp.get("/accounts")
@has_access("realestate_acc:ledger")
def list_accounts():
	property_id = request.args.get("property_id")
	accounts = _run(_svc.list_accounts(_tenant(), property_id))
	return _ok([a.model_dump() for a in accounts])


@bp.post("/accounts")
@has_access("realestate_acc:ledger")
def create_account():
	try:
		payload = AccountCreate(**request.json, tenant_id=_tenant())
		record = _run(_svc.create_account(payload))
		return _ok(record.model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/accounts/<account_id>")
@has_access("realestate_acc:ledger")
def get_account(account_id: str):
	record = _run(_svc.get_account(account_id, _tenant()))
	return _ok(record.model_dump()) if record else _err("not found", 404)


@bp.put("/accounts/<account_id>")
@has_access("realestate_acc:ledger")
def update_account(account_id: str):
	try:
		updates = AccountUpdate(**request.json)
		record = _run(_svc.update_account(account_id, _tenant(), updates))
		return _ok(record.model_dump()) if record else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Journal Entries ───────────────────────────────────────────────────────────

@bp.get("/journals")
@has_access("realestate_acc:journals")
def list_journals():
	period = request.args.get("period")
	property_id = request.args.get("property_id")
	journals = _run(_svc.list_journals(_tenant(), period, property_id))
	return _ok([j.model_dump() for j in journals])


@bp.post("/journals")
@has_access("realestate_acc:journals")
def create_journal():
	try:
		payload = JournalEntryCreate(**request.json, tenant_id=_tenant())
		record = _run(_svc.create_journal_entry(payload))
		return _ok(record.model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/journals/<journal_id>/approve")
@has_access("realestate_acc:journals")
def approve_journal(journal_id: str):
	approved_by = request.json.get("approved_by", "unknown")
	record = _run(_svc.approve_journal_entry(journal_id, _tenant(), approved_by))
	return _ok(record.model_dump()) if record else _err("not found", 404)


@bp.post("/journals/<journal_id>/post")
@has_access("realestate_acc:journals")
def post_journal(journal_id: str):
	try:
		record = _run(_svc.post_journal_entry(journal_id, _tenant()))
		return _ok(record.model_dump()) if record else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/journals/<journal_id>/reverse")
@has_access("realestate_acc:journals")
def reverse_journal(journal_id: str):
	try:
		reversed_by = request.json.get("reversed_by", "unknown")
		record = _run(_svc.reverse_journal_entry(journal_id, _tenant(), reversed_by))
		return _ok(record.model_dump(), 201) if record else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Service Charges ───────────────────────────────────────────────────────────

@bp.get("/service-charges")
@has_access("realestate_acc:service_charges")
def list_service_charges():
	charges = _run(_svc.list_service_charges(_tenant(), request.args.get("property_id"), request.args.get("period")))
	return _ok([c.model_dump() for c in charges])


@bp.post("/service-charges")
@has_access("realestate_acc:service_charges")
def create_service_charge():
	try:
		payload = ServiceChargeCreate(**request.json, tenant_id=_tenant())
		record = _run(_svc.raise_service_charge(payload))
		return _ok(record.model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/service-charges/<charge_id>/approve")
@has_access("realestate_acc:service_charges")
def approve_service_charge(charge_id: str):
	approved_by = request.json.get("approved_by", "unknown")
	record = _run(_svc.approve_service_charge(charge_id, _tenant(), approved_by))
	return _ok(record.model_dump()) if record else _err("not found", 404)


# ── CAM Reconciliation ────────────────────────────────────────────────────────

@bp.get("/cam")
@has_access("realestate_acc:cam")
def list_cam():
	cams = _run(_svc.list_cam_reconciliations(_tenant(), request.args.get("property_id")))
	return _ok([c.model_dump() for c in cams])


@bp.post("/cam")
@has_access("realestate_acc:cam")
def create_cam():
	try:
		payload = CamReconciliationCreate(**request.json, tenant_id=_tenant())
		record = _run(_svc.start_cam_reconciliation(payload))
		return _ok(record.model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/cam/<cam_id>/approve")
@has_access("realestate_acc:cam")
def approve_cam(cam_id: str):
	approved_by = request.json.get("approved_by", "unknown")
	record = _run(_svc.approve_cam_reconciliation(cam_id, _tenant(), approved_by))
	return _ok(record.model_dump()) if record else _err("not found", 404)


@bp.post("/cam/<cam_id>/settle")
@has_access("realestate_acc:cam")
def settle_cam(cam_id: str):
	try:
		record = _run(_svc.settle_cam_reconciliation(cam_id, _tenant()))
		return _ok(record.model_dump()) if record else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── IFRS 16 ───────────────────────────────────────────────────────────────────

@bp.post("/ifrs16")
@has_access("realestate_acc:ifrs16")
def create_ifrs16():
	try:
		payload = Ifrs16ScheduleCreate(**request.json, tenant_id=_tenant())
		record = _run(_svc.generate_ifrs16_schedule(payload))
		return _ok(record.model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/ifrs16/<schedule_id>")
@has_access("realestate_acc:ifrs16")
def get_ifrs16(schedule_id: str):
	record = _run(_svc.get_ifrs16_schedule(schedule_id, _tenant()))
	return _ok(record.model_dump()) if record else _err("not found", 404)


# ── Revenue Schedules ─────────────────────────────────────────────────────────

@bp.post("/revenue")
@has_access("realestate_acc:revenue")
def create_revenue_schedule():
	try:
		payload = RevenueScheduleCreate(**request.json, tenant_id=_tenant())
		record = _run(_svc.create_revenue_schedule(payload))
		return _ok(record.model_dump(), 201)
	except Exception as e:
		return _err(str(e))


# ── Period Management ─────────────────────────────────────────────────────────

@bp.post("/periods")
@has_access("realestate_acc:period_close")
def open_period():
	try:
		payload = AccountingPeriodCreate(**request.json, tenant_id=_tenant())
		record = _run(_svc.open_period(payload))
		return _ok(record.model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/periods/<period_id>/close")
@has_access("realestate_acc:period_close")
def close_period(period_id: str):
	try:
		data = request.json
		record = _run(_svc.close_period(period_id, _tenant(), data["closed_by"], data["second_approver"]))
		return _ok(record.model_dump()) if record else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Tenant Statements ─────────────────────────────────────────────────────────

@bp.post("/statements")
@has_access("realestate_acc:statements")
def generate_statement():
	try:
		payload = TenantStatementCreate(**request.json, tenant_id=_tenant())
		record = _run(_svc.generate_tenant_statement(payload))
		return _ok(record.model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/statements/<statement_id>")
@has_access("realestate_acc:statements")
def get_statement(statement_id: str):
	record = _run(_svc.get_tenant_statement(statement_id, _tenant()))
	return _ok(record.model_dump()) if record else _err("not found", 404)


# ── Reports ───────────────────────────────────────────────────────────────────

@bp.get("/reports/trial-balance")
@has_access("realestate_acc:reports")
def trial_balance():
	period = request.args.get("period", "")
	if not period:
		return _err("period is required")
	report = _run(_svc.generate_trial_balance(_tenant(), period))
	return _ok(report)
