"""Payroll UI Views — Flask Blueprint.

Renders HTML via Jinja2 templates (or returns JSON screen models when
templates are not present). Every view enforces tenant isolation via
request context.

URL prefix: /hcm/payroll
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any

from flask import (
	Blueprint,
	Response,
	abort,
	jsonify,
	redirect,
	render_template,
	request,
	url_for,
)

try:
	from .service import PayrollManagementService
except ImportError:
	from service import PayrollManagementService  # type: ignore


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

ui = Blueprint(
	"payroll_ui",
	__name__,
	url_prefix="/hcm/payroll",
	template_folder="templates/payroll",
)

_svc = PayrollManagementService()


def _get_svc() -> PayrollManagementService:
	return _svc


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-Id") or request.args.get("tenant_id", "default")


def _render(template: str, **ctx: Any) -> Response:
	"""Try template render; fall back to JSON screen model for API consumers."""
	try:
		return render_template(template, **ctx)
	except Exception:
		return jsonify(ctx)


# ---------------------------------------------------------------------------
# Navigation config (injected into every template context)
# ---------------------------------------------------------------------------

NAVIGATION = [
	{"name": "Dashboard",          "route": "payroll_ui.dashboard",         "icon": "layout-dashboard"},
	{"name": "Employees",          "route": "payroll_ui.list_employees",     "icon": "users"},
	{"name": "Pay Periods",        "route": "payroll_ui.list_periods",       "icon": "calendar-days"},
	{"name": "Pay Groups",         "route": "payroll_ui.list_pay_groups",    "icon": "layers"},
	{"name": "Profiles",           "route": "payroll_ui.list_profiles",      "icon": "id-card"},
	{"name": "Payroll Runs",       "route": "payroll_ui.list_runs",          "icon": "calculator"},
	{"name": "Payslips",           "route": "payroll_ui.list_payslips",      "icon": "file-text"},
	{"name": "Leave",              "route": "payroll_ui.list_leave",         "icon": "beach"},
	{"name": "Overtime",           "route": "payroll_ui.list_overtime",      "icon": "clock"},
	{"name": "Advances",           "route": "payroll_ui.list_advances",      "icon": "wallet"},
	{"name": "Settlements",        "route": "payroll_ui.list_settlements",   "icon": "handshake"},
	{"name": "GL Entries",         "route": "payroll_ui.list_gl_entries",    "icon": "landmark"},
	{"name": "Reports",            "route": "payroll_ui.reports_index",      "icon": "bar-chart"},
	{"name": "Settings",           "route": "payroll_ui.settings",           "icon": "settings"},
]


def _base_ctx(screen: str) -> dict[str, Any]:
	tenant = _tenant()
	return {
		"screen": screen,
		"tenant_id": tenant,
		"navigation": NAVIGATION,
		"now": datetime.utcnow().isoformat(timespec="seconds"),
	}


# ===========================================================================
# Dashboard
# ===========================================================================

@ui.get("/")
@ui.get("/dashboard")
def dashboard() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	ctx = _base_ctx("dashboard")
	ctx["summary"] = svc.dashboard_summary(tenant)
	ctx["open_periods"] = [
		p for p in svc.periods.values()
		if p.get("tenant_id") == tenant and p.get("status") == "open"
	]
	ctx["pending_runs"] = [
		r for r in svc.runs.values()
		if r.get("tenant_id") == tenant and r.get("status") in ("calculated", "under_review")
	]
	ctx["recent_runs"] = sorted(
		[r for r in svc.runs.values() if r.get("tenant_id") == tenant],
		key=lambda r: r.get("created_at", ""),
		reverse=True,
	)[:5]
	ctx["kpis"] = _build_kpis(svc, tenant)
	return _render("dashboard.html", **ctx)


def _build_kpis(svc: PayrollManagementService, tenant: str) -> dict[str, Any]:
	runs = [r for r in svc.runs.values() if r.get("tenant_id") == tenant]
	posted = [r for r in runs if r.get("status") in ("posted", "paid")]
	total_net = sum(r.get("totals", {}).get("net", 0) for r in posted)
	total_employees = len([
		p for p in svc.employee_pay_profiles.values()
		if p.get("tenant_id") == tenant and p.get("status") == "active"
	])
	return {
		"total_runs": len(runs),
		"posted_runs": len(posted),
		"total_net_paid": round(total_net, 2),
		"active_employees": total_employees,
		"pending_approvals": len([r for r in runs if not r.get("approved_by") and r.get("status") == "calculated"]),
	}


# ===========================================================================
# Employees
# ===========================================================================

@ui.get("/employees")
def list_employees() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "employees"):
		svc.employees = {}
	ctx = _base_ctx("employees")
	ctx["employees"] = [
		e for e in svc.employees.values()
		if e.get("tenant_id") == tenant and not e.get("is_deleted")
	]
	ctx["columns"] = [
		"employee_number", "full_name", "department_name",
		"employment_type", "basic_salary", "currency", "country",
	]
	return _render("list_employees.html", **ctx)


@ui.get("/employees/create")
def create_employee_form() -> Response:
	ctx = _base_ctx("create_employee")
	ctx["form"] = {}
	return _render("create_employee.html", **ctx)


@ui.post("/employees/create")
def create_employee_submit() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "employees"):
		svc.employees = {}
	data = request.form.to_dict()
	data["tenant_id"] = tenant
	from uuid6 import uuid7
	emp_id = str(uuid7())
	record = {**data, "id": emp_id, "is_active": True, "is_deleted": False,
			  "created_at": svc._now(), "updated_at": svc._now()}
	svc.employees[emp_id] = record
	return redirect(url_for("payroll_ui.detail_employee", employee_id=emp_id))


@ui.get("/employees/<employee_id>")
def detail_employee(employee_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "employees"):
		svc.employees = {}
	emp = svc.employees.get(employee_id)
	if not emp or emp.get("tenant_id") != tenant:
		abort(404)
	ctx = _base_ctx("employee_detail")
	ctx["employee"] = emp
	ctx["leave_balances"] = [
		b for b in getattr(svc, "leave_balances", {}).values()
		if b.get("employee_id") == employee_id
	]
	ctx["advances"] = [
		a for a in svc.salary_advances.values()
		if a.get("employee_id") == employee_id
	]
	return _render("detail_employee.html", **ctx)


@ui.get("/employees/<employee_id>/edit")
def edit_employee(employee_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "employees"):
		svc.employees = {}
	emp = svc.employees.get(employee_id)
	if not emp or emp.get("tenant_id") != tenant:
		abort(404)
	ctx = _base_ctx("edit_employee")
	ctx["employee"] = emp
	return _render("edit_employee.html", **ctx)


@ui.post("/employees/<employee_id>/edit")
def edit_employee_submit(employee_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "employees"):
		svc.employees = {}
	emp = svc.employees.get(employee_id)
	if not emp or emp.get("tenant_id") != tenant:
		abort(404)
	data = request.form.to_dict()
	data.pop("id", None)
	data.pop("tenant_id", None)
	emp.update(data)
	emp["updated_at"] = svc._now()
	return redirect(url_for("payroll_ui.detail_employee", employee_id=employee_id))


# ===========================================================================
# Pay Periods
# ===========================================================================

@ui.get("/periods")
def list_periods() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	ctx = _base_ctx("periods")
	ctx["periods"] = sorted(
		[p for p in svc.periods.values() if p.get("tenant_id") == tenant],
		key=lambda p: p.get("start_date", ""),
		reverse=True,
	)
	ctx["columns"] = ["name", "frequency", "start_date", "end_date", "pay_date", "status", "currency"]
	return _render("list_periods.html", **ctx)


@ui.get("/periods/create")
def create_period_form() -> Response:
	ctx = _base_ctx("create_period")
	ctx["form"] = {"pay_frequency": "monthly", "currency": "KES", "country": "KE"}
	return _render("create_period.html", **ctx)


@ui.post("/periods/create")
def create_period_submit() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	data = request.form.to_dict()
	record = svc.create_payroll_period(
		data.get("period_code", ""),
		tenant,
		data.get("name", data.get("period_code", "")),
		data.get("pay_frequency", "monthly"),
		data["start_date"],
		data["end_date"],
		data["pay_date"],
		data.get("currency", "KES"),
	)
	return redirect(url_for("payroll_ui.detail_period", period_id=record["id"]))


@ui.get("/periods/<period_id>")
def detail_period(period_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	p = svc.periods.get(period_id)
	if not p or p.get("tenant_id") != tenant:
		abort(404)
	runs = [r for r in svc.runs.values() if r.get("period_id") == period_id and r.get("tenant_id") == tenant]
	ctx = _base_ctx("period_detail")
	ctx["period"] = p
	ctx["runs"] = runs
	return _render("detail_period.html", **ctx)


@ui.get("/periods/<period_id>/edit")
def edit_period(period_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	p = svc.periods.get(period_id)
	if not p or p.get("tenant_id") != tenant:
		abort(404)
	ctx = _base_ctx("edit_period")
	ctx["period"] = p
	return _render("edit_period.html", **ctx)


# ===========================================================================
# Pay Groups
# ===========================================================================

@ui.get("/pay-groups")
def list_pay_groups() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	ctx = _base_ctx("pay_groups")
	ctx["pay_groups"] = [g for g in svc.pay_groups.values() if g.get("tenant_id") == tenant]
	ctx["columns"] = ["code", "name", "frequency", "currency", "country", "status"]
	return _render("list_pay_groups.html", **ctx)


@ui.get("/pay-groups/create")
def create_pay_group_form() -> Response:
	ctx = _base_ctx("create_pay_group")
	ctx["form"] = {"pay_frequency": "monthly", "currency": "KES", "country": "KE"}
	return _render("create_pay_group.html", **ctx)


@ui.post("/pay-groups/create")
def create_pay_group_submit() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	data = request.form.to_dict()
	record = svc.create_pay_group(
		data.get("id", ""),
		tenant,
		data["code"],
		data["name"],
		data.get("pay_frequency", "monthly"),
		data.get("currency", "KES"),
		data.get("country", "KE"),
		data.get("owner_id", "system"),
	)
	return redirect(url_for("payroll_ui.list_pay_groups"))


@ui.get("/pay-groups/<group_id>")
def detail_pay_group(group_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	g = svc.pay_groups.get(group_id)
	if not g or g.get("tenant_id") != tenant:
		abort(404)
	profiles = [p for p in svc.employee_pay_profiles.values() if p.get("pay_group_id") == group_id]
	ctx = _base_ctx("pay_group_detail")
	ctx["pay_group"] = g
	ctx["profiles"] = profiles
	return _render("detail_pay_group.html", **ctx)


# ===========================================================================
# Employee Pay Profiles
# ===========================================================================

@ui.get("/profiles")
def list_profiles() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	ctx = _base_ctx("profiles")
	ctx["profiles"] = [
		p for p in svc.employee_pay_profiles.values()
		if p.get("tenant_id") == tenant
	]
	ctx["columns"] = ["employee_id", "pay_group_id", "payment_method", "currency", "base_pay", "status"]
	return _render("list_profiles.html", **ctx)


@ui.get("/profiles/create")
def create_profile_form() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	ctx = _base_ctx("create_profile")
	ctx["pay_groups"] = [g for g in svc.pay_groups.values() if g.get("tenant_id") == tenant]
	ctx["form"] = {"currency": "KES", "payment_method": "bank_transfer"}
	return _render("create_profile.html", **ctx)


@ui.post("/profiles/create")
def create_profile_submit() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	data = request.form.to_dict()
	record = svc.create_employee_pay_profile(
		data.get("id", ""),
		tenant,
		data["employee_id"],
		data["pay_group_id"],
		data.get("payment_method", "bank_transfer"),
		data.get("tax_id", ""),
		data.get("currency", "KES"),
		float(data.get("base_pay", 0)),
		data.get("reviewed_by"),
		basic_pay=float(data.get("basic_pay", data.get("base_pay", 0))),
		hire_date=data.get("hire_date"),
		bank_account=data.get("bank_account"),
	)
	return redirect(url_for("payroll_ui.list_profiles"))


@ui.get("/profiles/<profile_id>")
def detail_profile(profile_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	p = svc.employee_pay_profiles.get(profile_id)
	if not p or p.get("tenant_id") != tenant:
		abort(404)
	ctx = _base_ctx("profile_detail")
	ctx["profile"] = p
	ctx["pay_group"] = svc.pay_groups.get(p.get("pay_group_id", ""))
	return _render("detail_profile.html", **ctx)


@ui.get("/profiles/<profile_id>/edit")
def edit_profile(profile_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	p = svc.employee_pay_profiles.get(profile_id)
	if not p or p.get("tenant_id") != tenant:
		abort(404)
	ctx = _base_ctx("edit_profile")
	ctx["profile"] = p
	ctx["pay_groups"] = [g for g in svc.pay_groups.values() if g.get("tenant_id") == tenant]
	return _render("edit_profile.html", **ctx)


# ===========================================================================
# Payroll Runs
# ===========================================================================

@ui.get("/runs")
def list_runs() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	ctx = _base_ctx("runs")
	ctx["runs"] = sorted(
		[r for r in svc.runs.values() if r.get("tenant_id") == tenant],
		key=lambda r: r.get("created_at", ""),
		reverse=True,
	)
	ctx["columns"] = ["period_id", "pay_group_id", "status", "employee_count",
					  "total_gross", "total_net", "approved_by", "created_at"]
	return _render("list_runs.html", **ctx)


@ui.get("/runs/create")
def create_run_form() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	ctx = _base_ctx("create_run")
	ctx["periods"] = [p for p in svc.periods.values() if p.get("tenant_id") == tenant and p.get("status") == "open"]
	ctx["pay_groups"] = [g for g in svc.pay_groups.values() if g.get("tenant_id") == tenant]
	return _render("create_run.html", **ctx)


@ui.post("/runs/create")
def create_run_submit() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	data = request.form.to_dict()
	record = svc.start_payroll_run(
		data.get("id", ""),
		tenant,
		data["period_id"],
		data["pay_group_id"],
		data.get("initiated_by", "system"),
	)
	return redirect(url_for("payroll_ui.detail_run", run_id=record["id"]))


@ui.get("/runs/<run_id>")
def detail_run(run_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	r = svc.runs.get(run_id)
	if not r or r.get("tenant_id") != tenant:
		abort(404)
	period = svc.periods.get(r.get("period_id", ""))
	pay_group = svc.pay_groups.get(r.get("pay_group_id", ""))
	line_items = [l for l in svc.line_items.values() if l.get("run_id") == run_id]
	ctx = _base_ctx("run_detail")
	ctx["run"] = r
	ctx["period"] = period
	ctx["pay_group"] = pay_group
	ctx["line_items"] = line_items
	ctx["payslip_lines"] = r.get("payslip_lines", [])
	ctx["can_approve"] = r.get("status") in ("calculated",)
	ctx["can_post"] = r.get("status") == "approved"
	ctx["can_reverse"] = r.get("status") in ("posted", "paid")
	return _render("detail_run.html", **ctx)


@ui.post("/runs/<run_id>/calculate")
def calculate_run(run_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	r = svc.runs.get(run_id)
	if not r or r.get("tenant_id") != tenant:
		abort(404)
	_run(svc.run_payroll(r["period_id"], tenant, r["pay_group_id"], r.get("initiated_by", "system")))
	return redirect(url_for("payroll_ui.detail_run", run_id=run_id))


@ui.post("/runs/<run_id>/approve")
def approve_run(run_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	approved_by = request.form.get("approved_by") or request.headers.get("X-Actor-Id", "system")
	svc.approve_payroll_run(run_id, tenant, approved_by)
	return redirect(url_for("payroll_ui.detail_run", run_id=run_id))


@ui.post("/runs/<run_id>/post")
def post_run(run_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	posted_by = request.form.get("posted_by") or request.headers.get("X-Actor-Id", "system")
	svc.post_payroll_run(run_id, tenant, posted_by)
	return redirect(url_for("payroll_ui.detail_run", run_id=run_id))


@ui.post("/runs/<run_id>/reverse")
def reverse_run(run_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	r = svc.runs.get(run_id)
	if not r or r.get("tenant_id") != tenant:
		abort(404)
	reason = request.form.get("reason", "")
	if not reason:
		abort(400)
	r["status"] = "reversed"
	r["reversed_by"] = request.form.get("reversed_by") or request.headers.get("X-Actor-Id", "system")
	r["reversal_reason"] = reason
	r["updated_at"] = svc._now()
	return redirect(url_for("payroll_ui.detail_run", run_id=run_id))


# ===========================================================================
# Payslips
# ===========================================================================

@ui.get("/payslips")
def list_payslips() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	run_id = request.args.get("run_id")
	ctx = _base_ctx("payslips")
	ctx["payslips"] = [
		p for p in svc.payslips.values()
		if p.get("tenant_id") == tenant
		and (not run_id or p.get("run_id") == run_id)
	]
	ctx["run_id"] = run_id
	ctx["columns"] = ["employee_id", "run_id", "pay_date", "net_pay", "status"]
	return _render("list_payslips.html", **ctx)


@ui.get("/payslips/<run_id>/<employee_id>")
def view_payslip(run_id: str, employee_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	try:
		payslip = _run(svc.generate_payslip(employee_id, run_id, tenant_id=tenant))
	except Exception as exc:
		abort(404)
	ctx = _base_ctx("payslip")
	ctx["payslip"] = payslip
	ctx["run"] = svc.runs.get(run_id, {})
	return _render("view_payslip.html", **ctx)


# ===========================================================================
# Leave
# ===========================================================================

@ui.get("/leave")
def list_leave() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "leave_balances"):
		svc.leave_balances = {}
	ctx = _base_ctx("leave")
	ctx["leave_balances"] = [
		b for b in svc.leave_balances.values()
		if b.get("tenant_id") == tenant
	]
	ctx["columns"] = ["employee_id", "leave_type", "year", "entitled_days", "taken_days", "balance"]
	return _render("list_leave.html", **ctx)


@ui.get("/leave/create")
def create_leave_form() -> Response:
	ctx = _base_ctx("create_leave")
	ctx["form"] = {"year": date.today().year, "entitled_days": 21}
	return _render("create_leave.html", **ctx)


@ui.post("/leave/create")
def create_leave_submit() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "leave_balances"):
		svc.leave_balances = {}
	data = request.form.to_dict()
	from uuid6 import uuid7
	lb_id = str(uuid7())
	entitled = float(data.get("entitled_days", 0))
	taken = float(data.get("taken_days", 0))
	carried = float(data.get("carried_forward", 0))
	record = {
		**data,
		"id": lb_id,
		"tenant_id": tenant,
		"entitled_days": entitled,
		"taken_days": taken,
		"carried_forward": carried,
		"balance": entitled + carried - taken,
		"encashed_days": 0,
		"encashed_amount": 0,
		"created_at": svc._now(),
		"updated_at": svc._now(),
	}
	svc.leave_balances[lb_id] = record
	return redirect(url_for("payroll_ui.list_leave"))


@ui.get("/leave/<balance_id>")
def detail_leave(balance_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "leave_balances"):
		svc.leave_balances = {}
	lb = svc.leave_balances.get(balance_id)
	if not lb or lb.get("tenant_id") != tenant:
		abort(404)
	ctx = _base_ctx("leave_detail")
	ctx["leave_balance"] = lb
	return _render("detail_leave.html", **ctx)


# ===========================================================================
# Overtime
# ===========================================================================

@ui.get("/overtime")
def list_overtime() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "overtime_records"):
		svc.overtime_records = {}
	ctx = _base_ctx("overtime")
	ctx["overtime_records"] = [
		o for o in svc.overtime_records.values()
		if o.get("tenant_id") == tenant
	]
	ctx["columns"] = ["employee_id", "run_id", "overtime_hours", "overtime_type", "overtime_pay"]
	return _render("list_overtime.html", **ctx)


@ui.get("/overtime/create")
def create_overtime_form() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	ctx = _base_ctx("create_overtime")
	ctx["runs"] = [r for r in svc.runs.values() if r.get("tenant_id") == tenant and r.get("status") in ("draft", "calculated")]
	ctx["form"] = {"overtime_type": "time_and_half", "regular_hours": 173.33}
	return _render("create_overtime.html", **ctx)


@ui.post("/overtime/create")
def create_overtime_submit() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	if not hasattr(svc, "overtime_records"):
		svc.overtime_records = {}
	data = request.form.to_dict()
	result = _run(svc.calculate_overtime(
		data["employee_id"],
		float(data.get("regular_hours", 173.33)),
		float(data["overtime_hours"]),
		data.get("overtime_type", "time_and_half"),
		tenant_id=tenant,
	))
	from uuid6 import uuid7
	ot_id = str(uuid7())
	record = {**result, "id": ot_id, "tenant_id": tenant, "run_id": data.get("run_id"), "created_at": svc._now()}
	svc.overtime_records[ot_id] = record
	return redirect(url_for("payroll_ui.list_overtime"))


# ===========================================================================
# Salary Advances
# ===========================================================================

@ui.get("/advances")
def list_advances() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	ctx = _base_ctx("advances")
	ctx["advances"] = [
		a for a in svc.salary_advances.values()
		if a.get("tenant_id") == tenant
	]
	ctx["columns"] = ["employee_id", "amount", "monthly_instalment", "balance", "status", "approved_by"]
	return _render("list_advances.html", **ctx)


@ui.get("/advances/create")
def create_advance_form() -> Response:
	ctx = _base_ctx("create_advance")
	ctx["form"] = {}
	return _render("create_advance.html", **ctx)


@ui.post("/advances/create")
def create_advance_submit() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	data = request.form.to_dict()
	record = svc.create_salary_advance(
		data.get("id", ""),
		tenant,
		data["employee_id"],
		float(data["amount"]),
		float(data["monthly_instalment"]),
		data.get("approved_by", "system"),
	)
	return redirect(url_for("payroll_ui.list_advances"))


@ui.get("/advances/<advance_id>")
def detail_advance(advance_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	a = svc.salary_advances.get(advance_id)
	if not a or a.get("tenant_id") != tenant:
		abort(404)
	ctx = _base_ctx("advance_detail")
	ctx["advance"] = a
	return _render("detail_advance.html", **ctx)


# ===========================================================================
# Final Settlements
# ===========================================================================

@ui.get("/settlements")
def list_settlements() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	ctx = _base_ctx("settlements")
	ctx["settlements"] = [s for s in svc.terminal_benefits.values() if s.get("tenant_id") == tenant]
	ctx["columns"] = ["employee_id", "termination_date", "reason", "total_terminal_pay", "status"]
	return _render("list_settlements.html", **ctx)


@ui.get("/settlements/create")
def create_settlement_form() -> Response:
	ctx = _base_ctx("create_settlement")
	ctx["form"] = {"reason": "resignation"}
	ctx["reasons"] = ["resignation", "redundancy", "retirement", "dismissal", "death", "contract_end"]
	return _render("create_settlement.html", **ctx)


@ui.post("/settlements/create")
def create_settlement_submit() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	data = request.form.to_dict()
	result = _run(svc.calculate_terminal_benefits(
		data["employee_id"],
		data["termination_date"],
		data.get("reason", "resignation"),
		tenant_id=tenant,
		leave_days_accrued=float(data.get("leave_days_accrued", 0)),
	))
	return redirect(url_for("payroll_ui.detail_settlement", employee_id=data["employee_id"]))


@ui.get("/settlements/<employee_id>")
def detail_settlement(employee_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	settlements = [
		s for s in svc.terminal_benefits.values()
		if s.get("tenant_id") == tenant and s.get("employee_id") == employee_id
	]
	if not settlements:
		abort(404)
	ctx = _base_ctx("settlement_detail")
	ctx["settlement"] = settlements[-1]
	return _render("detail_settlement.html", **ctx)


# ===========================================================================
# GL Entries
# ===========================================================================

@ui.get("/gl-entries")
def list_gl_entries() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	run_id = request.args.get("run_id")
	ctx = _base_ctx("gl_entries")
	ctx["gl_entries"] = [
		e for e in svc.gl_entries.values()
		if e.get("tenant_id") == tenant
		and (not run_id or e.get("run_id") == run_id)
	]
	ctx["columns"] = ["run_id", "total_debits", "total_credits", "balanced", "status", "posted_at"]
	return _render("list_gl_entries.html", **ctx)


@ui.get("/gl-entries/<entry_id>")
def detail_gl_entry(entry_id: str) -> Response:
	svc = _get_svc()
	tenant = _tenant()
	e = svc.gl_entries.get(entry_id)
	if not e or e.get("tenant_id") != tenant:
		abort(404)
	ctx = _base_ctx("gl_entry_detail")
	ctx["gl_entry"] = e
	return _render("detail_gl_entry.html", **ctx)


# ===========================================================================
# Reports
# ===========================================================================

@ui.get("/reports")
def reports_index() -> Response:
	ctx = _base_ctx("reports")
	ctx["report_types"] = [
		{"name": "Payroll Summary",    "route": "payroll_ui.report_summary",      "description": "Run totals — gross, deductions, net"},
		{"name": "Variance Report",    "route": "payroll_ui.report_variance",      "description": "Salary movement vs prior period"},
		{"name": "P9 Form (Kenya)",    "route": "payroll_ui.report_p9",            "description": "Annual PAYE declaration (KRA P9)"},
		{"name": "NSSF Schedule",      "route": "payroll_ui.report_nssf",          "description": "Monthly NSSF contribution schedule"},
		{"name": "NHIF Schedule",      "route": "payroll_ui.report_nhif",          "description": "Monthly NHIF contribution schedule"},
		{"name": "Statutory Returns",  "route": "payroll_ui.report_statutory",     "description": "All statutory return schedules"},
		{"name": "Bank Transfer File", "route": "payroll_ui.report_bank_file",     "description": "EFT/bulk payment file"},
		{"name": "Cost Centre",        "route": "payroll_ui.report_cost_centre",   "description": "Payroll cost by cost centre"},
	]
	return _render("reports_index.html", **ctx)


@ui.get("/reports/summary")
def report_summary() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	run_id = request.args.get("run_id")
	ctx = _base_ctx("report_summary")
	ctx["runs"] = [r for r in svc.runs.values() if r.get("tenant_id") == tenant]
	if run_id:
		r = svc.runs.get(run_id)
		period = svc.periods.get((r or {}).get("period_id", ""), {})
		ctx["report"] = {
			"run_id": run_id,
			"period_code": period.get("name", ""),
			"pay_date": period.get("pay_date", ""),
			"status": (r or {}).get("status"),
			"employee_count": (r or {}).get("employee_count", 0),
			"totals": (r or {}).get("totals", {}),
			"generated_at": svc._now(),
		}
	return _render("report_summary.html", **ctx)


@ui.get("/reports/variance")
def report_variance() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	run_id = request.args.get("run_id")
	ctx = _base_ctx("report_variance")
	ctx["runs"] = [r for r in svc.runs.values() if r.get("tenant_id") == tenant]
	if run_id:
		ctx["report"] = _run(svc.payroll_variance_report(run_id, tenant_id=tenant))
	return _render("report_variance.html", **ctx)


@ui.get("/reports/p9")
def report_p9() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	employee_id = request.args.get("employee_id")
	year = int(request.args.get("year", date.today().year))
	ctx = _base_ctx("report_p9")
	ctx["year"] = year
	if employee_id:
		try:
			ctx["report"] = _run(svc.generate_p9_form(employee_id, year, tenant_id=tenant))
		except Exception as exc:
			ctx["error"] = str(exc)
	return _render("report_p9.html", **ctx)


@ui.get("/reports/nssf")
def report_nssf() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	period_id = request.args.get("period_id")
	country = request.args.get("country", "KE")
	ctx = _base_ctx("report_nssf")
	ctx["periods"] = [p for p in svc.periods.values() if p.get("tenant_id") == tenant]
	if period_id:
		try:
			ctx["report"] = _run(svc.nssf_schedules_report(period_id, country, tenant_id=tenant))
		except Exception as exc:
			ctx["error"] = str(exc)
	return _render("report_nssf.html", **ctx)


@ui.get("/reports/nhif")
def report_nhif() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	period_id = request.args.get("period_id")
	country = request.args.get("country", "KE")
	ctx = _base_ctx("report_nhif")
	ctx["periods"] = [p for p in svc.periods.values() if p.get("tenant_id") == tenant]
	if period_id:
		try:
			returns = _run(svc.generate_statutory_returns(period_id, country, tenant_id=tenant))
			ctx["report"] = returns.get("nhif_schedule", {})
		except Exception as exc:
			ctx["error"] = str(exc)
	return _render("report_nhif.html", **ctx)


@ui.get("/reports/statutory")
def report_statutory() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	period_id = request.args.get("period_id")
	country = request.args.get("country", "KE")
	ctx = _base_ctx("report_statutory")
	ctx["periods"] = [p for p in svc.periods.values() if p.get("tenant_id") == tenant]
	ctx["country"] = country
	if period_id:
		try:
			ctx["report"] = _run(svc.generate_statutory_returns(period_id, country, tenant_id=tenant))
		except Exception as exc:
			ctx["error"] = str(exc)
	return _render("report_statutory.html", **ctx)


@ui.get("/reports/bank-file")
def report_bank_file() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	run_id = request.args.get("run_id")
	fmt = request.args.get("format", "KCB_EFT")
	ctx = _base_ctx("report_bank_file")
	ctx["runs"] = [r for r in svc.runs.values() if r.get("tenant_id") == tenant and r.get("status") in ("approved", "posted", "paid")]
	ctx["format"] = fmt
	if run_id:
		try:
			ctx["report"] = _run(svc.bank_transfer_file(run_id, fmt, tenant_id=tenant))
		except Exception as exc:
			ctx["error"] = str(exc)
	return _render("report_bank_file.html", **ctx)


@ui.get("/reports/cost-centre")
def report_cost_centre() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	run_id = request.args.get("run_id")
	ctx = _base_ctx("report_cost_centre")
	ctx["runs"] = [r for r in svc.runs.values() if r.get("tenant_id") == tenant]
	if run_id:
		r = svc.runs.get(run_id)
		if r and r.get("tenant_id") == tenant:
			costs: dict[str, float] = {}
			for line in r.get("payslip_lines", []):
				profile = svc.employee_pay_profiles.get(line.get("profile_id", ""))
				cc = (profile or {}).get("cost_center", "UNASSIGNED")
				costs[cc] = costs.get(cc, 0.0) + line.get("gross", 0.0)
			ctx["report"] = {
				"run_id": run_id,
				"cost_by_center": costs,
				"generated_at": svc._now(),
			}
	return _render("report_cost_centre.html", **ctx)


# ===========================================================================
# Settings
# ===========================================================================

@ui.get("/settings")
def settings() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	ctx = _base_ctx("settings")
	ctx["config"] = {}
	if hasattr(svc, "configs"):
		configs = [c for c in svc.configs.values() if c.get("tenant_id") == tenant]
		ctx["config"] = configs[0] if configs else {}
	ctx["supported_countries"] = ["KE", "TZ", "UG", "RW", "ZM", "GH", "NG", "ZA", "ET", "ZW"]
	ctx["supported_currencies"] = ["KES", "TZS", "UGX", "RWF", "ZMW", "GHS", "NGN", "ZAR", "USD", "EUR", "GBP"]
	return _render("settings.html", **ctx)


@ui.post("/settings")
def settings_submit() -> Response:
	svc = _get_svc()
	tenant = _tenant()
	data = request.form.to_dict()
	if not hasattr(svc, "configs"):
		svc.configs = {}
	existing = next(
		(c for c in svc.configs.values() if c.get("tenant_id") == tenant),
		None,
	)
	if existing:
		existing.update({k: v for k, v in data.items() if k not in ("id", "tenant_id")})
		existing["updated_at"] = svc._now()
	else:
		from uuid6 import uuid7
		cfg_id = str(uuid7())
		svc.configs[cfg_id] = {**data, "id": cfg_id, "tenant_id": tenant, "created_at": svc._now()}
	return redirect(url_for("payroll_ui.settings"))


# ---------------------------------------------------------------------------
# Backwards-compat shims
# ---------------------------------------------------------------------------

class PayrollPeriodModelView:
	pass


class PayrollRunModelView:
	pass


class EmployeePayrollModelView:
	pass


class PayComponentModelView:
	pass


class PayrollDashboardView:
	pass


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return [{"name": n["name"], "route": n["route"]} for n in NAVIGATION]


def dashboard_model(svc: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return {
		"screen": "dashboard",
		"tenant_id": tenant_id,
		"navigation": NAVIGATION,
		"summary": svc.dashboard_summary(tenant_id),
	}


# ---------------------------------------------------------------------------
# Screen model helpers (module-level, used by contract tests and API helpers)
# ---------------------------------------------------------------------------

def _records_model(svc: PayrollManagementService, tenant_id: str, screen: str, collection: str, columns: list[str]) -> dict[str, Any]:
	return {
		"screen": screen,
		"tenant_id": tenant_id,
		"navigation": NAVIGATION,
		"records": svc.list_records(collection, tenant_id),
		"columns": columns,
	}


def period_model(svc: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records_model(svc, tenant_id, "periods", "periods",
		["name", "frequency", "start_date", "end_date", "pay_date", "currency", "status"])


def pay_group_model(svc: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records_model(svc, tenant_id, "pay_groups", "pay_groups",
		["code", "name", "frequency", "currency", "country", "owner_id", "status"])


def profile_model(svc: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records_model(svc, tenant_id, "profiles", "employee_pay_profiles",
		["employee_id", "pay_group_id", "payment_method", "currency", "base_pay", "status"])


def component_model(svc: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records_model(svc, tenant_id, "components", "components",
		["code", "name", "component_type", "currency", "taxable", "status"])


def run_model(svc: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records_model(svc, tenant_id, "runs", "runs",
		["period_id", "pay_group_id", "initiated_by", "approved_by", "posted_by", "status"])


def payslip_model(svc: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	return _records_model(svc, tenant_id, "payslips", "payslips",
		["run_id", "employee_id", "privacy_basis", "net_pay", "status"])


def agent_workbench_model(svc: PayrollManagementService, tenant_id: str) -> dict[str, Any]:
	model = _records_model(svc, tenant_id, "agents", "agents",
		["name", "runtime", "role", "scope", "status"])
	model["actions"] = ["review_run", "review_tax", "review_payment", "review_variance", "review_employee_query"]
	return model
