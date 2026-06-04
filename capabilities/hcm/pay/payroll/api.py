"""Payroll REST API — Flask Blueprint.

All endpoints enforce tenant isolation via X-Tenant-Id header.
Responses follow: {"data": ..., "meta": {...}} or {"error": ..., "code": ...}

URL prefix: /api/v1/payroll
"""
from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal
from functools import wraps
from typing import Any

from flask import Blueprint, Response, jsonify, request

try:
	from .service import PayrollManagementService
except ImportError:
	from service import PayrollManagementService  # type: ignore


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

bp = Blueprint("payroll_api", __name__, url_prefix="/api/v1/payroll")

# Process-local service instance (swap for DB-backed in production)
_svc = PayrollManagementService()


def _get_svc() -> PayrollManagementService:
	return _svc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro: Any) -> Any:
	"""Run an async coroutine from sync Flask context."""
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _tenant() -> str:
	t = request.headers.get("X-Tenant-Id") or request.args.get("tenant_id", "default")
	return t


def _ok(data: Any, status: int = 200, meta: dict | None = None) -> Response:
	body: dict[str, Any] = {"data": data}
	if meta:
		body["meta"] = meta
	return jsonify(body), status


def _err(message: str, code: int = 400, details: Any = None) -> Response:
	body: dict[str, Any] = {"error": message, "code": code}
	if details:
		body["details"] = details
	return jsonify(body), code


def _paginate(items: list, page: int, per_page: int) -> tuple[list, dict]:
	total = len(items)
	start = (page - 1) * per_page
	end = start + per_page
	return items[start:end], {
		"page": page,
		"per_page": per_page,
		"total": total,
		"pages": max(1, (total + per_page - 1) // per_page),
	}


def handle_errors(fn):
	@wraps(fn)
	def wrapper(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except PermissionError as e:
			return _err(str(e), 403)
		except KeyError as e:
			return _err(f"Not found: {e}", 404)
		except (ValueError, AssertionError) as e:
			return _err(str(e), 400)
		except Exception as e:
			return _err(f"Internal error: {e}", 500)
	return wrapper


# ===========================================================================
# Health / status
# ===========================================================================

@bp.get("/health")
@handle_errors
def health():
	svc = _get_svc()
	summary = svc.dashboard_summary(_tenant())
	return _ok({"status": "ok", "summary": summary})


# ===========================================================================
# Payroll Config
# ===========================================================================

@bp.get("/config")
@handle_errors
def get_config():
	svc = _get_svc()
	tenant = _tenant()
	configs = [c for c in svc.__dict__.get("configs", {}).values() if c.get("tenant_id") == tenant]
	return _ok(configs)


@bp.post("/config")
@handle_errors
def create_config():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tenant
	if not hasattr(svc, "configs"):
		svc.configs = {}
	from uuid import uuid4
	cfg_id = str(uuid4())
	record = {**body, "id": cfg_id, "created_at": svc._now()}
	svc.configs[cfg_id] = record
	return _ok(record, 201)


# ===========================================================================
# Employees
# ===========================================================================

@bp.get("/employees")
@handle_errors
def list_employees():
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "employees"):
		svc.employees = {}
	page = int(request.args.get("page", 1))
	per_page = int(request.args.get("per_page", 50))
	dept = request.args.get("department")
	emp_type = request.args.get("employment_type")
	active_only = request.args.get("active_only", "true").lower() == "true"

	items = [
		e for e in svc.employees.values()
		if e.get("tenant_id") == tenant and not e.get("is_deleted")
		and (not active_only or e.get("is_active", True))
		and (not dept or e.get("department_id") == dept)
		and (not emp_type or e.get("employment_type") == emp_type)
	]
	page_items, meta = _paginate(items, page, per_page)
	return _ok(page_items, meta=meta)


@bp.post("/employees")
@handle_errors
def create_employee():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tenant
	if not hasattr(svc, "employees"):
		svc.employees = {}
	from uuid6 import uuid7
	emp_id = str(uuid7())
	record = {
		**body,
		"id": emp_id,
		"is_active": True,
		"is_deleted": False,
		"created_at": svc._now(),
		"updated_at": svc._now(),
	}
	svc.employees[emp_id] = record
	return _ok(record, 201)


@bp.get("/employees/<employee_id>")
@handle_errors
def get_employee(employee_id: str):
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "employees"):
		svc.employees = {}
	emp = svc.employees.get(employee_id)
	if not emp or emp.get("tenant_id") != tenant or emp.get("is_deleted"):
		return _err("Employee not found", 404)
	return _ok(emp)


@bp.put("/employees/<employee_id>")
@handle_errors
def update_employee(employee_id: str):
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "employees"):
		svc.employees = {}
	emp = svc.employees.get(employee_id)
	if not emp or emp.get("tenant_id") != tenant or emp.get("is_deleted"):
		return _err("Employee not found", 404)
	body = request.get_json(force=True) or {}
	body.pop("id", None)
	body.pop("tenant_id", None)
	emp.update(body)
	emp["updated_at"] = svc._now()
	return _ok(emp)


@bp.delete("/employees/<employee_id>")
@handle_errors
def delete_employee(employee_id: str):
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "employees"):
		svc.employees = {}
	emp = svc.employees.get(employee_id)
	if not emp or emp.get("tenant_id") != tenant:
		return _err("Employee not found", 404)
	emp["is_deleted"] = True
	emp["is_active"] = False
	emp["updated_at"] = svc._now()
	return _ok({"deleted": True, "id": employee_id})


# ===========================================================================
# Pay Periods
# ===========================================================================

@bp.get("/periods")
@handle_errors
def list_periods():
	svc = _get_svc()
	tenant = _tenant()
	page = int(request.args.get("page", 1))
	per_page = int(request.args.get("per_page", 50))
	status_filter = request.args.get("status")
	items = [
		p for p in svc.periods.values()
		if p.get("tenant_id") == tenant
		and (not status_filter or p.get("status") == status_filter)
	]
	page_items, meta = _paginate(items, page, per_page)
	return _ok(page_items, meta=meta)


@bp.post("/periods")
@handle_errors
def create_period():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	period_id = body.get("period_code", body.get("id", f"period-{svc._now()[:10]}"))
	record = svc.create_payroll_period(
		period_id,
		tenant,
		body.get("name", period_id),
		body.get("pay_frequency", body.get("frequency", "monthly")),
		body["start_date"],
		body["end_date"],
		body["pay_date"],
		body.get("currency", "KES"),
	)
	return _ok(record, 201)


@bp.get("/periods/<period_id>")
@handle_errors
def get_period(period_id: str):
	svc = _get_svc()
	tenant = _tenant()
	p = svc.periods.get(period_id)
	if not p or p.get("tenant_id") != tenant:
		return _err("Period not found", 404)
	return _ok(p)


@bp.put("/periods/<period_id>")
@handle_errors
def update_period(period_id: str):
	svc = _get_svc()
	tenant = _tenant()
	p = svc.periods.get(period_id)
	if not p or p.get("tenant_id") != tenant:
		return _err("Period not found", 404)
	body = request.get_json(force=True) or {}
	for k, v in body.items():
		if k not in ("id", "tenant_id"):
			p[k] = v
	p["updated_at"] = svc._now()
	return _ok(p)


@bp.post("/periods/<period_id>/lock")
@handle_errors
def lock_period(period_id: str):
	svc = _get_svc()
	tenant = _tenant()
	p = svc.periods.get(period_id)
	if not p or p.get("tenant_id") != tenant:
		return _err("Period not found", 404)
	p["status"] = "locked"
	p["updated_at"] = svc._now()
	return _ok(p)


@bp.post("/periods/<period_id>/close")
@handle_errors
def close_period(period_id: str):
	svc = _get_svc()
	tenant = _tenant()
	p = svc.periods.get(period_id)
	if not p or p.get("tenant_id") != tenant:
		return _err("Period not found", 404)
	p["status"] = "closed"
	p["updated_at"] = svc._now()
	return _ok(p)


# ===========================================================================
# Pay Groups
# ===========================================================================

@bp.get("/pay-groups")
@handle_errors
def list_pay_groups():
	svc = _get_svc()
	tenant = _tenant()
	items = [g for g in svc.pay_groups.values() if g.get("tenant_id") == tenant]
	return _ok(items)


@bp.post("/pay-groups")
@handle_errors
def create_pay_group():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	record = svc.create_pay_group(
		body.get("id", ""),
		tenant,
		body["code"],
		body["name"],
		body.get("pay_frequency", body.get("frequency", "monthly")),
		body.get("currency", "KES"),
		body.get("country", "KE"),
		body.get("owner_id", "system"),
	)
	return _ok(record, 201)


@bp.get("/pay-groups/<group_id>")
@handle_errors
def get_pay_group(group_id: str):
	svc = _get_svc()
	tenant = _tenant()
	g = svc.pay_groups.get(group_id)
	if not g or g.get("tenant_id") != tenant:
		return _err("Pay group not found", 404)
	return _ok(g)


# ===========================================================================
# Employee Pay Profiles
# ===========================================================================

@bp.get("/profiles")
@handle_errors
def list_profiles():
	svc = _get_svc()
	tenant = _tenant()
	page = int(request.args.get("page", 1))
	per_page = int(request.args.get("per_page", 50))
	group_id = request.args.get("pay_group_id")
	items = [
		p for p in svc.employee_pay_profiles.values()
		if p.get("tenant_id") == tenant
		and (not group_id or p.get("pay_group_id") == group_id)
	]
	page_items, meta = _paginate(items, page, per_page)
	return _ok(page_items, meta=meta)


@bp.post("/profiles")
@handle_errors
def create_profile():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	record = svc.create_employee_pay_profile(
		body.get("id", ""),
		tenant,
		body["employee_id"],
		body["pay_group_id"],
		body.get("payment_method", "bank_transfer"),
		body.get("tax_id", body.get("tax_pin", "")),
		body.get("currency", "KES"),
		float(body.get("base_pay", body.get("basic_salary", 0))),
		body.get("reviewed_by"),
		basic_pay=float(body.get("basic_pay", body.get("base_pay", 0))),
		hire_date=body.get("hire_date"),
		bank_account=body.get("bank_account"),
	)
	return _ok(record, 201)


@bp.get("/profiles/<profile_id>")
@handle_errors
def get_profile(profile_id: str):
	svc = _get_svc()
	tenant = _tenant()
	p = svc.employee_pay_profiles.get(profile_id)
	if not p or p.get("tenant_id") != tenant:
		return _err("Profile not found", 404)
	return _ok(p)


@bp.put("/profiles/<profile_id>")
@handle_errors
def update_profile(profile_id: str):
	svc = _get_svc()
	tenant = _tenant()
	p = svc.employee_pay_profiles.get(profile_id)
	if not p or p.get("tenant_id") != tenant:
		return _err("Profile not found", 404)
	body = request.get_json(force=True) or {}
	for k, v in body.items():
		if k not in ("id", "tenant_id"):
			p[k] = v
	p["updated_at"] = svc._now()
	return _ok(p)


@bp.delete("/profiles/<profile_id>")
@handle_errors
def delete_profile(profile_id: str):
	svc = _get_svc()
	tenant = _tenant()
	p = svc.employee_pay_profiles.get(profile_id)
	if not p or p.get("tenant_id") != tenant:
		return _err("Profile not found", 404)
	p["status"] = "inactive"
	p["is_deleted"] = True
	p["updated_at"] = svc._now()
	return _ok({"deleted": True, "id": profile_id})


# ===========================================================================
# Payroll Runs
# ===========================================================================

@bp.get("/runs")
@handle_errors
def list_runs():
	svc = _get_svc()
	tenant = _tenant()
	page = int(request.args.get("page", 1))
	per_page = int(request.args.get("per_page", 50))
	status_filter = request.args.get("status")
	period_id = request.args.get("period_id")
	items = [
		r for r in svc.runs.values()
		if r.get("tenant_id") == tenant
		and (not status_filter or r.get("status") == status_filter)
		and (not period_id or r.get("period_id") == period_id)
	]
	page_items, meta = _paginate(items, page, per_page)
	return _ok(page_items, meta=meta)


@bp.post("/runs")
@handle_errors
def create_run():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	record = svc.start_payroll_run(
		body.get("id", ""),
		tenant,
		body["period_id"],
		body["pay_group_id"],
		body.get("initiated_by", request.headers.get("X-Actor-Id", "system")),
	)
	return _ok(record, 201)


@bp.get("/runs/<run_id>")
@handle_errors
def get_run(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	r = svc.runs.get(run_id)
	if not r or r.get("tenant_id") != tenant:
		return _err("Run not found", 404)
	return _ok(r)


@bp.post("/runs/<run_id>/calculate")
@handle_errors
def calculate_run(run_id: str):
	"""Full payroll calculation for all employees in the run's pay group."""
	svc = _get_svc()
	tenant = _tenant()
	r = svc.runs.get(run_id)
	if not r or r.get("tenant_id") != tenant:
		return _err("Run not found", 404)
	result = _run(svc.run_payroll(
		r["period_id"],
		tenant,
		r["pay_group_id"],
		r.get("initiated_by", "system"),
	))
	return _ok(result)


@bp.post("/runs/<run_id>/approve")
@handle_errors
def approve_run(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	approved_by = body.get("approved_by") or request.headers.get("X-Actor-Id", "system")
	record = svc.approve_payroll_run(run_id, tenant, approved_by)
	return _ok(record)


@bp.post("/runs/<run_id>/reject")
@handle_errors
def reject_run(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	r = svc.runs.get(run_id)
	if not r or r.get("tenant_id") != tenant:
		return _err("Run not found", 404)
	body = request.get_json(force=True) or {}
	r["status"] = "draft"
	r["rejection_reason"] = body.get("reason", "")
	r["rejected_by"] = body.get("rejected_by") or request.headers.get("X-Actor-Id", "system")
	r["updated_at"] = svc._now()
	return _ok(r)


@bp.post("/runs/<run_id>/post")
@handle_errors
def post_run(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	posted_by = body.get("posted_by") or request.headers.get("X-Actor-Id", "system")
	record = svc.post_payroll_run(run_id, tenant, posted_by)
	return _ok(record)


@bp.post("/runs/<run_id>/cancel")
@handle_errors
def cancel_run(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	r = svc.runs.get(run_id)
	if not r or r.get("tenant_id") != tenant:
		return _err("Run not found", 404)
	r["status"] = "cancelled"
	r["updated_at"] = svc._now()
	return _ok(r)


@bp.post("/runs/<run_id>/reverse")
@handle_errors
def reverse_run(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	r = svc.runs.get(run_id)
	if not r or r.get("tenant_id") != tenant:
		return _err("Run not found", 404)
	body = request.get_json(force=True) or {}
	reason = body.get("reason", "")
	if not reason:
		return _err("Reversal reason is required", 400)
	r["status"] = "reversed"
	r["reversed_by"] = body.get("reversed_by") or request.headers.get("X-Actor-Id", "system")
	r["reversal_reason"] = reason
	r["updated_at"] = svc._now()
	return _ok(r)


@bp.post("/runs/<run_id>/gl-post")
@handle_errors
def gl_post_run(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	result = _run(svc.gl_posting(run_id, tenant_id=tenant, gl_accounts=body.get("gl_accounts")))
	return _ok(result)


@bp.get("/runs/<run_id>/variance")
@handle_errors
def variance_report(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	compare_to = request.args.get("compare_to_run_id")
	result = _run(svc.payroll_variance_report(run_id, tenant_id=tenant, compare_to_run_id=compare_to))
	return _ok(result)


@bp.get("/runs/<run_id>/bank-file")
@handle_errors
def bank_file(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	fmt = request.args.get("format", "KCB_EFT")
	result = _run(svc.bank_transfer_file(run_id, fmt, tenant_id=tenant))
	return _ok(result)


# ===========================================================================
# Bonus Runs
# ===========================================================================

@bp.post("/runs/bonus")
@handle_errors
def create_bonus_run():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	result = _run(svc.process_bonus_payroll(
		body.get("bonus_type", "performance"),
		body.get("employee_ids", []),
		body.get("amounts", {}),
		body.get("tax_method", "aggregate"),
		tenant_id=tenant,
		period_id=body.get("period_id"),
	))
	return _ok(result, 201)


# ===========================================================================
# Payslip Lines
# ===========================================================================

@bp.get("/runs/<run_id>/lines")
@handle_errors
def list_run_lines(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	lines = [l for l in svc.line_items.values() if l.get("run_id") == run_id and l.get("tenant_id") == tenant]
	return _ok(lines)


@bp.post("/runs/<run_id>/lines")
@handle_errors
def add_line(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	record = svc.add_line_item(
		body.get("id", ""),
		tenant,
		run_id,
		body["profile_id"],
		body["component_id"],
		float(body["amount"]),
		body.get("reviewed_by"),
	)
	return _ok(record, 201)


# ===========================================================================
# Payslips
# ===========================================================================

@bp.get("/payslips")
@handle_errors
def list_payslips():
	svc = _get_svc()
	tenant = _tenant()
	page = int(request.args.get("page", 1))
	per_page = int(request.args.get("per_page", 50))
	run_id = request.args.get("run_id")
	employee_id = request.args.get("employee_id")
	items = [
		p for p in svc.payslips.values()
		if p.get("tenant_id") == tenant
		and (not run_id or p.get("run_id") == run_id)
		and (not employee_id or p.get("employee_id") == employee_id)
	]
	page_items, meta = _paginate(items, page, per_page)
	return _ok(page_items, meta=meta)


@bp.get("/payslips/<run_id>/<employee_id>")
@handle_errors
def get_payslip(run_id: str, employee_id: str):
	svc = _get_svc()
	tenant = _tenant()
	result = _run(svc.generate_payslip(employee_id, run_id, tenant_id=tenant))
	return _ok(result)


@bp.post("/payslips/<run_id>/bulk-email")
@handle_errors
def bulk_email_payslips(run_id: str):
	"""Stub — in production dispatches email jobs per employee."""
	svc = _get_svc()
	tenant = _tenant()
	r = svc.runs.get(run_id)
	if not r or r.get("tenant_id") != tenant:
		return _err("Run not found", 404)
	employee_count = r.get("employee_count", 0)
	return _ok({"queued": True, "run_id": run_id, "recipients": employee_count})


# ===========================================================================
# Tax Calculations
# ===========================================================================

@bp.post("/tax/calculate-paye")
@handle_errors
def calculate_paye():
	svc = _get_svc()
	body = request.get_json(force=True) or {}
	result = _run(svc.calculate_paye(
		float(body["gross_monthly"]),
		body.get("country", "KE"),
		allowances=body.get("allowances"),
		deductions=body.get("deductions"),
		ytd_gross=float(body.get("ytd_gross", 0)),
	))
	return _ok(result)


@bp.post("/tax/calculate-statutory")
@handle_errors
def calculate_statutory():
	svc = _get_svc()
	body = request.get_json(force=True) or {}
	result = _run(svc.calculate_statutory_deductions(
		body.get("employee", {}),
		float(body["gross"]),
		body.get("country", "KE"),
	))
	return _ok(result)


@bp.get("/tax/p9/<employee_id>/<int:year>")
@handle_errors
def p9_form(employee_id: str, year: int):
	svc = _get_svc()
	tenant = _tenant()
	result = _run(svc.generate_p9_form(employee_id, year, tenant_id=tenant))
	return _ok(result)


# ===========================================================================
# Statutory Returns
# ===========================================================================

@bp.get("/statutory-returns/<period_id>")
@handle_errors
def statutory_returns(period_id: str):
	svc = _get_svc()
	tenant = _tenant()
	country = request.args.get("country", "KE")
	result = _run(svc.generate_statutory_returns(period_id, country, tenant_id=tenant))
	return _ok(result)


@bp.get("/statutory-returns/<period_id>/nssf")
@handle_errors
def nssf_schedule(period_id: str):
	svc = _get_svc()
	tenant = _tenant()
	country = request.args.get("country", "KE")
	result = _run(svc.nssf_schedules_report(period_id, country, tenant_id=tenant))
	return _ok(result)


# ===========================================================================
# Leave Balances
# ===========================================================================

@bp.get("/leave")
@handle_errors
def list_leave():
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "leave_balances"):
		svc.leave_balances = {}
	employee_id = request.args.get("employee_id")
	year = request.args.get("year")
	items = [
		b for b in svc.leave_balances.values()
		if b.get("tenant_id") == tenant
		and (not employee_id or b.get("employee_id") == employee_id)
		and (not year or str(b.get("year")) == str(year))
	]
	return _ok(items)


@bp.post("/leave")
@handle_errors
def create_leave_balance():
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "leave_balances"):
		svc.leave_balances = {}
	body = request.get_json(force=True) or {}
	from uuid6 import uuid7
	lb_id = str(uuid7())
	record = {
		**body,
		"id": lb_id,
		"tenant_id": tenant,
		"balance": float(body.get("entitled_days", 0)) + float(body.get("carried_forward", 0)) - float(body.get("taken_days", 0)),
		"created_at": svc._now(),
		"updated_at": svc._now(),
	}
	svc.leave_balances[lb_id] = record
	return _ok(record, 201)


@bp.put("/leave/<balance_id>")
@handle_errors
def update_leave_balance(balance_id: str):
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "leave_balances"):
		svc.leave_balances = {}
	lb = svc.leave_balances.get(balance_id)
	if not lb or lb.get("tenant_id") != tenant:
		return _err("Leave balance not found", 404)
	body = request.get_json(force=True) or {}
	for k, v in body.items():
		if k not in ("id", "tenant_id"):
			lb[k] = v
	lb["balance"] = (float(lb.get("entitled_days", 0))
		+ float(lb.get("carried_forward", 0))
		- float(lb.get("taken_days", 0)))
	lb["updated_at"] = svc._now()
	return _ok(lb)


@bp.post("/leave/<employee_id>/encash")
@handle_errors
def encash_leave(employee_id: str):
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	result = _run(svc.calculate_leave_encashment(
		employee_id,
		body.get("leave_type", "annual"),
		float(body.get("days", 0)),
		tenant_id=tenant,
	))
	return _ok(result)


@bp.post("/leave/carry-forward")
@handle_errors
def carry_forward_leave():
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "leave_balances"):
		svc.leave_balances = {}
	body = request.get_json(force=True) or {}
	from_year = int(body.get("from_year", date.today().year - 1))
	to_year = int(body.get("to_year", date.today().year))
	max_carry = float(body.get("max_carry_days", 30))
	carried = []
	for lb in svc.leave_balances.values():
		if lb.get("tenant_id") != tenant or lb.get("year") != from_year:
			continue
		balance = float(lb.get("entitled_days", 0)) - float(lb.get("taken_days", 0))
		carry = min(balance, max_carry)
		if carry > 0:
			from uuid6 import uuid7
			new_id = str(uuid7())
			new_lb = {
				"id": new_id,
				"tenant_id": tenant,
				"employee_id": lb["employee_id"],
				"leave_type": lb["leave_type"],
				"year": to_year,
				"entitled_days": lb.get("entitled_days", 0),
				"taken_days": 0,
				"carried_forward": carry,
				"balance": float(lb.get("entitled_days", 0)) + carry,
				"encashed_days": 0,
				"encashed_amount": 0,
				"created_at": svc._now(),
				"updated_at": svc._now(),
			}
			svc.leave_balances[new_id] = new_lb
			carried.append({"employee_id": lb["employee_id"], "days_carried": carry, "new_balance_id": new_id})
	return _ok({"carried_count": len(carried), "details": carried})


# ===========================================================================
# Overtime
# ===========================================================================

@bp.get("/overtime")
@handle_errors
def list_overtime():
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "overtime_records"):
		svc.overtime_records = {}
	run_id = request.args.get("run_id")
	items = [
		o for o in svc.overtime_records.values()
		if o.get("tenant_id") == tenant
		and (not run_id or o.get("run_id") == run_id)
	]
	return _ok(items)


@bp.post("/overtime")
@handle_errors
def create_overtime():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	result = _run(svc.calculate_overtime(
		body["employee_id"],
		float(body.get("regular_hours", 173.33)),
		float(body["overtime_hours"]),
		body.get("overtime_type", "time_and_half"),
		tenant_id=tenant,
	))
	if not hasattr(svc, "overtime_records"):
		svc.overtime_records = {}
	from uuid6 import uuid7
	ot_id = str(uuid7())
	record = {**result, "id": ot_id, "tenant_id": tenant, "run_id": body.get("run_id"), "created_at": svc._now()}
	svc.overtime_records[ot_id] = record
	return _ok(record, 201)


# ===========================================================================
# Salary Advances
# ===========================================================================

@bp.get("/advances")
@handle_errors
def list_advances():
	svc = _get_svc()
	tenant = _tenant()
	employee_id = request.args.get("employee_id")
	active_only = request.args.get("active_only", "false").lower() == "true"
	items = [
		a for a in svc.salary_advances.values()
		if a.get("tenant_id") == tenant
		and (not employee_id or a.get("employee_id") == employee_id)
		and (not active_only or a.get("status") == "active")
	]
	return _ok(items)


@bp.post("/advances")
@handle_errors
def create_advance():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	record = svc.create_salary_advance(
		body.get("id", ""),
		tenant,
		body["employee_id"],
		float(body["amount"]),
		float(body["monthly_instalment"]),
		body.get("approved_by") or request.headers.get("X-Actor-Id", "system"),
	)
	return _ok(record, 201)


@bp.get("/advances/<advance_id>")
@handle_errors
def get_advance(advance_id: str):
	svc = _get_svc()
	tenant = _tenant()
	a = svc.salary_advances.get(advance_id)
	if not a or a.get("tenant_id") != tenant:
		return _err("Advance not found", 404)
	return _ok(a)


@bp.post("/advances/<advance_id>/deduct")
@handle_errors
def deduct_advance(advance_id: str):
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	result = _run(svc.apply_salary_advance_deduction(
		body["employee_id"],
		advance_id,
		body["run_id"],
		tenant_id=tenant,
	))
	return _ok(result)


# ===========================================================================
# Garnishments
# ===========================================================================

@bp.get("/garnishments")
@handle_errors
def list_garnishments():
	svc = _get_svc()
	tenant = _tenant()
	employee_id = request.args.get("employee_id")
	items = [
		g for g in svc.garnishments.values()
		if g.get("tenant_id") == tenant
		and (not employee_id or g.get("employee_id") == employee_id)
	]
	return _ok(items)


@bp.post("/garnishments")
@handle_errors
def process_garnishment():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	result = _run(svc.process_garnishment(
		body["employee_id"],
		body["garnishment_order"],
		body["run_id"],
		tenant_id=tenant,
	))
	return _ok(result, 201)


# ===========================================================================
# Final Settlement / Terminal Benefits
# ===========================================================================

@bp.get("/settlements")
@handle_errors
def list_settlements():
	svc = _get_svc()
	tenant = _tenant()
	items = [s for s in svc.terminal_benefits.values() if s.get("tenant_id") == tenant]
	return _ok(items)


@bp.post("/settlements")
@handle_errors
def create_settlement():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	result = _run(svc.calculate_terminal_benefits(
		body["employee_id"],
		body["termination_date"],
		body.get("reason", "resignation"),
		tenant_id=tenant,
		leave_days_accrued=float(body.get("leave_days_accrued", 0)),
	))
	return _ok(result, 201)


@bp.get("/settlements/<employee_id>")
@handle_errors
def get_settlement(employee_id: str):
	svc = _get_svc()
	tenant = _tenant()
	settlements = [
		s for s in svc.terminal_benefits.values()
		if s.get("tenant_id") == tenant and s.get("employee_id") == employee_id
	]
	if not settlements:
		return _err("Settlement not found", 404)
	return _ok(settlements[-1])


# ===========================================================================
# GL Entries
# ===========================================================================

@bp.get("/gl-entries")
@handle_errors
def list_gl_entries():
	svc = _get_svc()
	tenant = _tenant()
	run_id = request.args.get("run_id")
	items = [
		e for e in svc.gl_entries.values()
		if e.get("tenant_id") == tenant
		and (not run_id or e.get("run_id") == run_id)
	]
	return _ok(items)


# ===========================================================================
# Expatriate Tax
# ===========================================================================

@bp.post("/tax/expatriate")
@handle_errors
def expatriate_tax():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	result = _run(svc.expatriate_tax_calculation(
		body["employee_id"],
		body.get("period", str(date.today())),
		tenant_id=tenant,
		home_country=body.get("home_country", "GB"),
		host_country=body.get("host_country", "KE"),
		company_bearing_tax=body.get("company_bearing_tax", True),
	))
	return _ok(result)


@bp.post("/tax/salary-sacrifice")
@handle_errors
def salary_sacrifice():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	result = _run(svc.salary_sacrifice_pension(
		body["employee_id"],
		body["amount_or_pct"],
		tenant_id=tenant,
		is_percentage=body.get("is_percentage", False),
	))
	return _ok(result)


# ===========================================================================
# Mid-month proration
# ===========================================================================

@bp.post("/proration/mid-hire")
@handle_errors
def mid_hire_proration():
	svc = _get_svc()
	tenant = _tenant()
	body = request.get_json(force=True) or {}
	result = _run(svc.mid_month_hire_calculation(
		body["employee_id"],
		body["hire_date"],
		body["period"],
		tenant_id=tenant,
	))
	return _ok(result)


# ===========================================================================
# Reports
# ===========================================================================

@bp.get("/reports/summary/<run_id>")
@handle_errors
def payroll_summary_report(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	r = svc.runs.get(run_id)
	if not r or r.get("tenant_id") != tenant:
		return _err("Run not found", 404)
	period = svc.periods.get(r.get("period_id", "")) or {}
	totals = r.get("totals", {})
	return _ok({
		"run_id": run_id,
		"period_code": period.get("name", ""),
		"pay_date": period.get("pay_date", ""),
		"status": r.get("status"),
		"employee_count": r.get("employee_count", 0),
		"total_gross": totals.get("gross", 0),
		"total_deductions": totals.get("deductions", 0),
		"total_taxes": totals.get("taxes", 0),
		"total_net": totals.get("net", 0),
		"generated_at": svc._now(),
	})


@bp.get("/reports/variance/<run_id>")
@handle_errors
def variance_report_get(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	compare_to = request.args.get("compare_to_run_id")
	result = _run(svc.payroll_variance_report(run_id, tenant_id=tenant, compare_to_run_id=compare_to))
	return _ok(result)


@bp.get("/reports/p9/<employee_id>/<int:year>")
@handle_errors
def p9_report(employee_id: str, year: int):
	svc = _get_svc()
	tenant = _tenant()
	result = _run(svc.generate_p9_form(employee_id, year, tenant_id=tenant))
	return _ok(result)


@bp.get("/reports/statutory/<period_id>")
@handle_errors
def statutory_report(period_id: str):
	svc = _get_svc()
	tenant = _tenant()
	country = request.args.get("country", "KE")
	result = _run(svc.generate_statutory_returns(period_id, country, tenant_id=tenant))
	return _ok(result)


@bp.get("/reports/bank-file/<run_id>")
@handle_errors
def bank_file_report(run_id: str):
	svc = _get_svc()
	tenant = _tenant()
	fmt = request.args.get("format", "KCB_EFT")
	result = _run(svc.bank_transfer_file(run_id, fmt, tenant_id=tenant))
	return _ok(result)


@bp.get("/reports/cost-center/<run_id>")
@handle_errors
def cost_center_report(run_id: str):
	"""Aggregate payroll cost by cost center for a run."""
	svc = _get_svc()
	tenant = _tenant()
	r = svc.runs.get(run_id)
	if not r or r.get("tenant_id") != tenant:
		return _err("Run not found", 404)
	costs: dict[str, float] = {}
	for line in r.get("payslip_lines", []):
		profile = svc.employee_pay_profiles.get(line.get("profile_id", ""))
		cc = (profile or {}).get("cost_center", "UNASSIGNED")
		costs[cc] = costs.get(cc, 0.0) + line.get("gross", 0.0)
	return _ok({"run_id": run_id, "cost_by_center": costs, "generated_at": svc._now()})


# ===========================================================================
# Dashboard
# ===========================================================================

@bp.get("/dashboard")
@handle_errors
def dashboard():
	svc = _get_svc()
	tenant = _tenant()
	summary = svc.dashboard_summary(tenant)
	return _ok(summary)


# ===========================================================================
# Audit Events
# ===========================================================================

@bp.get("/audit")
@handle_errors
def audit_events():
	svc = _get_svc()
	tenant = _tenant()
	page = int(request.args.get("page", 1))
	per_page = int(request.args.get("per_page", 100))
	events = svc.audit_events(tenant)
	page_items, meta = _paginate(events, page, per_page)
	return _ok(page_items, meta=meta)


# ---------------------------------------------------------------------------
# Backwards-compat shims — kept so old import paths don't break
# ---------------------------------------------------------------------------

class PayrollPeriodRestApi:
	pass


class PayrollRunRestApi:
	pass


class EmployeePayrollRestApi:
	pass


class PayComponentRestApi:
	pass


def service() -> PayrollManagementService:
	return _get_svc()


def register_api_endpoints(*_: Any, **__: Any) -> None:
	return None


def register_payroll_agent(payload: dict[str, Any]) -> dict[str, Any]:
	"""Module-level helper — delegates to service.register_payroll_agent."""
	svc = _get_svc()
	return svc.register_payroll_agent(
		payload["tenant_id"],
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("scope", "review payroll operations"),
	)


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	svc = _get_svc()
	return {"ok": True, "tenant_id": tenant_id, "summary": svc.dashboard_summary(tenant_id)}


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	svc = _get_svc()
	return svc.create_record(
		str(payload.get("id", "payroll-period")),
		str(payload.get("tenant_id") or "default"),
		payload,
		str(payload.get("status") or "open"),
	)


def list_records(collection: str | None = None, tenant_id: str = "default") -> list[dict[str, Any]]:
	return _get_svc().list_records(collection, tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return _get_svc().dashboard_summary(tenant_id)
