"""
Time & Attendance — Flask Blueprint UI Views

Serves HTML pages for the TAT capability workbench.
Each route returns a dict suitable for template rendering or JSON
when the Accept header requests application/json.

url_prefix: /hcm/time-attendance

Copyright © 2025 Datacraft. Author: Nyimbi Odero
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from flask import Blueprint, Response, jsonify, render_template_string, request

ui = Blueprint(
	"tat_ui",
	__name__,
	url_prefix="/hcm/time-attendance",
	template_folder="templates",
	static_folder="static",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro: Any) -> Any:
	"""Run an async coroutine from a sync Flask view."""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				return pool.submit(asyncio.run, coro).result(timeout=30)
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _svc() -> Any:
	"""Build service from request context."""
	from .service import TimeAttendanceService

	tenant_id = request.headers.get("X-Tenant-Id") or request.args.get("tenant_id") or "default"
	actor_id = request.headers.get("X-Actor-Id") or request.args.get("actor_id") or "system"

	try:
		from .database import get_db_session
		db = get_db_session()
	except Exception:
		db = _NullDB()

	return TimeAttendanceService(db, tenant_id, actor_id)


class _NullDB:
	async def execute(self, *a: Any, **kw: Any) -> None:
		raise RuntimeError("No DB configured")
	async def fetch(self, *a: Any, **kw: Any) -> list:
		raise RuntimeError("No DB configured")
	async def fetchrow(self, *a: Any, **kw: Any) -> None:
		raise RuntimeError("No DB configured")


def _wants_json() -> bool:
	accept = request.headers.get("Accept", "")
	return "application/json" in accept


def _respond(data: dict[str, Any], template_name: str | None = None) -> Response:
	if _wants_json() or not template_name:
		return jsonify(data)
	try:
		return render_template_string(_STUB_TEMPLATE, **data, view_name=template_name)
	except Exception:
		return jsonify(data)


# Minimal stub template — replace with real Jinja2 templates in production
_STUB_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title | default('Time & Attendance') }} — Datacraft APG</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #F7F8FA; color: #172033; }
  nav  { background: #255E56; color: #fff; padding: 0.75rem 1.5rem; display: flex; gap: 1.5rem; }
  nav a { color: #fff; text-decoration: none; font-size: 0.9rem; }
  main { padding: 1.5rem; }
  h1   { margin: 0 0 1rem; font-size: 1.5rem; }
  pre  { background: #fff; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.8rem; }
  .card { background: #fff; border-radius: 8px; padding: 1rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); margin-bottom: 1rem; }
  .kpi  { display: flex; gap: 1rem; flex-wrap: wrap; }
  .kpi .card { flex: 1; min-width: 140px; text-align: center; }
  .kpi .card h2 { font-size: 2rem; color: #255E56; margin: 0; }
  .kpi .card p  { margin: 0; font-size: 0.8rem; color: #52606D; }
</style>
</head>
<body>
<nav>
  <strong>⏱ Time &amp; Attendance</strong>
  <a href="/hcm/time-attendance/dashboard">Dashboard</a>
  <a href="/hcm/time-attendance/policies">Policies</a>
  <a href="/hcm/time-attendance/schedules">Schedules</a>
  <a href="/hcm/time-attendance/time-entries">Entries</a>
  <a href="/hcm/time-attendance/timesheets">Timesheets</a>
  <a href="/hcm/time-attendance/leave">Leave</a>
  <a href="/hcm/time-attendance/exceptions">Exceptions</a>
</nav>
<main>
  <h1>{{ title | default(view_name | title) }}</h1>
  {% if kpis %}
  <div class="kpi">
    {% for kpi in kpis %}
    <div class="card"><h2>{{ kpi.value }}</h2><p>{{ kpi.label }}</p></div>
    {% endfor %}
  </div>
  {% endif %}
  {% if records is defined %}
  <div class="card">
    <p>{{ records | length }} record(s)</p>
    <pre>{{ records | tojson(indent=2) }}</pre>
  </div>
  {% endif %}
  {% if detail is defined %}
  <div class="card"><pre>{{ detail | tojson(indent=2) }}</pre></div>
  {% endif %}
</main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@ui.get("/")
@ui.get("/dashboard")
def dashboard() -> Response:
	"""
	Main T&A dashboard with KPI tiles.

	KPIs: clocked-in now, pending timesheets, pending leaves, open exceptions,
	shifts today, hours this week.
	"""
	try:
		svc = _svc()
		summary = _run(svc.dashboard_summary())
	except Exception:
		summary = {}

	kpis = [
		{"label": "Clocked In Now", "value": summary.get("clocked_in_now", "—"), "icon": "timer"},
		{"label": "Shifts Today", "value": summary.get("shifts_today", "—"), "icon": "calendar"},
		{"label": "Pending Timesheets", "value": summary.get("pending_timesheets", "—"), "icon": "clipboard"},
		{"label": "Pending Leaves", "value": summary.get("pending_leaves", "—"), "icon": "calendar-minus"},
		{"label": "Open Exceptions", "value": summary.get("open_exceptions", "—"), "icon": "alert"},
		{"label": "Hours This Week", "value": summary.get("hours_this_week", "—"), "icon": "clock"},
	]
	data = {
		"title": "Time & Attendance Dashboard",
		"tenant_id": request.headers.get("X-Tenant-Id", "default"),
		"kpis": kpis,
		"summary": summary,
	}
	return _respond(data, "dashboard")


# ---------------------------------------------------------------------------
# Time Policies
# ---------------------------------------------------------------------------

@ui.get("/policies")
def list_policies() -> Response:
	"""List all time policies."""
	try:
		records = _run(_svc().list_time_policies(
			limit=int(request.args.get("limit", 50)),
			offset=int(request.args.get("offset", 0)),
		))
	except Exception as exc:
		records = []

	return _respond({
		"title": "Time Policies",
		"records": records,
		"create_url": "/hcm/time-attendance/policies/create",
	}, "policies")


@ui.get("/policies/create")
def create_policy_form() -> Response:
	"""Render the policy creation form."""
	return _respond({
		"title": "Create Time Policy",
		"form_action": "/hcm/time-attendance/api/v1/policies",
		"form_method": "POST",
	}, "policy_form")


@ui.get("/policies/<policy_id>")
def detail_policy(policy_id: str) -> Response:
	"""Policy detail view."""
	try:
		detail = _run(_svc().get_time_policy(policy_id))
	except Exception as exc:
		detail = {"error": str(exc)}
	return _respond({"title": "Policy Detail", "detail": detail}, "policy_detail")


@ui.get("/policies/<policy_id>/edit")
def edit_policy(policy_id: str) -> Response:
	"""Policy edit form."""
	try:
		detail = _run(_svc().get_time_policy(policy_id))
	except Exception as exc:
		detail = {"error": str(exc)}
	return _respond({
		"title": "Edit Policy",
		"detail": detail,
		"form_action": f"/hcm/time-attendance/api/v1/policies/{policy_id}",
		"form_method": "PUT",
	}, "policy_form")


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

@ui.get("/schedules")
def list_schedules() -> Response:
	"""List shift schedules."""
	try:
		records = _run(_svc().list_shift_schedules(
			department_id=request.args.get("department_id"),
			limit=int(request.args.get("limit", 50)),
			offset=int(request.args.get("offset", 0)),
		))
	except Exception:
		records = []
	return _respond({"title": "Shift Schedules", "records": records}, "schedules")


@ui.get("/schedules/<schedule_id>")
def detail_schedule(schedule_id: str) -> Response:
	try:
		detail = _run(_svc().get_shift_schedule(schedule_id))
	except Exception as exc:
		detail = {"error": str(exc)}
	return _respond({"title": "Schedule Detail", "detail": detail}, "schedule_detail")


# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------

@ui.get("/shifts")
def list_shifts() -> Response:
	"""Shift board view."""
	svc = _svc()
	try:
		records = _run(svc.list_shifts(
			employee_id=request.args.get("employee_id"),
			from_date=date.fromisoformat(request.args["from_date"]) if request.args.get("from_date") else None,
			to_date=date.fromisoformat(request.args["to_date"]) if request.args.get("to_date") else None,
			limit=int(request.args.get("limit", 100)),
			offset=int(request.args.get("offset", 0)),
		))
	except Exception:
		records = []
	return _respond({"title": "Shifts", "records": records}, "shifts")


# ---------------------------------------------------------------------------
# Time Entries
# ---------------------------------------------------------------------------

@ui.get("/time-entries")
def list_entries() -> Response:
	"""Time entry workbench."""
	svc = _svc()
	try:
		records = _run(svc.list_time_entries(
			employee_id=request.args.get("employee_id"),
			from_date=date.fromisoformat(request.args["from_date"]) if request.args.get("from_date") else None,
			to_date=date.fromisoformat(request.args["to_date"]) if request.args.get("to_date") else None,
			status=request.args.get("status"),
			limit=int(request.args.get("limit", 100)),
			offset=int(request.args.get("offset", 0)),
		))
	except Exception:
		records = []
	return _respond({"title": "Time Entries", "records": records}, "time_entries")


@ui.get("/time-entries/<entry_id>")
def detail_entry(entry_id: str) -> Response:
	try:
		detail = _run(_svc().get_time_entry(entry_id))
	except Exception as exc:
		detail = {"error": str(exc)}
	return _respond({"title": "Time Entry Detail", "detail": detail}, "entry_detail")


# ---------------------------------------------------------------------------
# Timesheets
# ---------------------------------------------------------------------------

@ui.get("/timesheets")
def list_timesheets() -> Response:
	"""Timesheet approval queue."""
	svc = _svc()
	try:
		records = _run(svc.list_timesheets(
			employee_id=request.args.get("employee_id"),
			status=request.args.get("status"),
			limit=int(request.args.get("limit", 50)),
			offset=int(request.args.get("offset", 0)),
		))
	except Exception:
		records = []
	return _respond({"title": "Timesheets", "records": records}, "timesheets")


@ui.get("/timesheets/<timesheet_id>")
def detail_timesheet(timesheet_id: str) -> Response:
	try:
		detail = _run(_svc().get_timesheet(timesheet_id))
	except Exception as exc:
		detail = {"error": str(exc)}
	return _respond({"title": "Timesheet Detail", "detail": detail}, "timesheet_detail")


# ---------------------------------------------------------------------------
# Leave
# ---------------------------------------------------------------------------

@ui.get("/leave")
def list_leave() -> Response:
	"""Leave request queue."""
	svc = _svc()
	try:
		records = _run(svc.list_leave_requests(
			employee_id=request.args.get("employee_id"),
			leave_type=request.args.get("leave_type"),
			status=request.args.get("status"),
			limit=int(request.args.get("limit", 50)),
			offset=int(request.args.get("offset", 0)),
		))
	except Exception:
		records = []
	return _respond({"title": "Leave Requests", "records": records}, "leave")


@ui.get("/leave/create")
def create_leave_form() -> Response:
	"""Leave request form."""
	return _respond({
		"title": "Request Leave",
		"leave_types": [
			"vacation", "sick", "personal", "maternity", "paternity", "parental",
			"bereavement", "jury_duty", "military", "sabbatical", "unpaid",
			"fmla", "toil", "comp_time", "study",
		],
		"form_action": "/hcm/time-attendance/api/v1/leave",
		"form_method": "POST",
	}, "leave_form")


@ui.get("/leave/<request_id>")
def detail_leave(request_id: str) -> Response:
	try:
		detail = _run(_svc().get_leave_request(request_id))
	except Exception as exc:
		detail = {"error": str(exc)}
	return _respond({"title": "Leave Request Detail", "detail": detail}, "leave_detail")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

@ui.get("/exceptions")
def list_exceptions() -> Response:
	"""Attendance exception centre."""
	svc = _svc()
	try:
		records = _run(svc.list_exceptions(
			employee_id=request.args.get("employee_id"),
			status=request.args.get("status"),
			severity=request.args.get("severity"),
			limit=int(request.args.get("limit", 50)),
			offset=int(request.args.get("offset", 0)),
		))
	except Exception:
		records = []
	return _respond({"title": "Attendance Exceptions", "records": records}, "exceptions")


# ---------------------------------------------------------------------------
# Payroll exports
# ---------------------------------------------------------------------------

@ui.get("/payroll-exports")
def list_exports() -> Response:
	"""Payroll export ledger."""
	try:
		records = _run(_svc().list_payroll_exports(
			limit=int(request.args.get("limit", 20)),
			offset=int(request.args.get("offset", 0)),
		))
	except Exception:
		records = []
	return _respond({"title": "Payroll Exports", "records": records}, "exports")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@ui.get("/reports")
def reports_index() -> Response:
	"""Reports landing page."""
	return _respond({
		"title": "Reports",
		"available_reports": [
			{"type": "daily_summary", "label": "Daily Attendance Summary", "icon": "chart-bar"},
			{"type": "overtime_report", "label": "Overtime Report", "icon": "clock-plus"},
			{"type": "leave_usage", "label": "Leave Usage Report", "icon": "calendar-minus"},
			{"type": "exception_report", "label": "Exception Report", "icon": "alert-triangle"},
		],
	}, "reports")


@ui.get("/reports/<report_type>")
def run_report(report_type: str) -> Response:
	"""Run and display a report."""
	from_date = date.fromisoformat(request.args["from_date"]) if request.args.get("from_date") else date.today().replace(day=1)
	to_date = date.fromisoformat(request.args["to_date"]) if request.args.get("to_date") else date.today()
	svc = _svc()
	try:
		result = _run(svc.generate_attendance_report(
			report_type=report_type,
			from_date=from_date,
			to_date=to_date,
			employee_id=request.args.get("employee_id"),
			department_id=request.args.get("department_id"),
		))
	except Exception as exc:
		result = {"error": str(exc)}

	return _respond({
		"title": report_type.replace("_", " ").title(),
		"report_type": report_type,
		"result": result,
		"from_date": from_date.isoformat(),
		"to_date": to_date.isoformat(),
	}, "report_view")


# ---------------------------------------------------------------------------
# Backward-compatible view model helpers (from original views.py)
# ---------------------------------------------------------------------------

def dashboard_model(service: Any, tenant_id: str) -> dict[str, Any]:
	"""Return a compact dashboard model for composed APG applications."""
	try:
		from .capability_contract import STREAMING
		summary = service.dashboard_summary(tenant_id) if hasattr(service, 'dashboard_summary') else {}
	except Exception:
		summary = {}
		STREAMING = {}

	from .capability_contract import STREAMING as _ST  # type: ignore[assignment]

	return {
		"title": "Time and Attendance",
		"tenant_id": tenant_id,
		"cards": [
			{"label": "Policies",   "value": summary.get("policy_count", 0),     "icon": "shield-check"},
			{"label": "Schedules",  "value": summary.get("schedule_count", 0),   "icon": "calendar-days"},
			{"label": "Entries",    "value": summary.get("time_entry_count", 0), "icon": "timer"},
			{"label": "Timesheets", "value": summary.get("timesheet_count", 0),  "icon": "clipboard-check"},
			{"label": "Exceptions", "value": summary.get("exception_count", 0),  "icon": "triangle-alert"},
		],
		"streaming": _ST,
	}


def policy_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Policies", "records": service.list_records(tenant_id, "policy")}


def schedule_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Schedules", "records": service.list_records(tenant_id, "schedule")}


def shift_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Shifts", "records": service.list_records(tenant_id, "shift")}


def time_entry_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Time Entries", "records": service.list_records(tenant_id, "time_entry")}


def timesheet_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Timesheets", "records": service.list_records(tenant_id, "timesheet")}


def leave_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Leave Requests", "records": service.list_records(tenant_id, "leave_request")}


def exception_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Exceptions", "records": service.list_records(tenant_id, "exception")}


def export_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Payroll Exports", "records": service.list_records(tenant_id, "payroll_export")}


def rules_model(contract: dict[str, Any]) -> dict[str, Any]:
	return {
		"name": "Attendance Rules",
		"rule_count": len(contract["rule_engine"]["rules"]),
		"rules": contract["rule_engine"]["rules"],
	}


def settings_model(contract: dict[str, Any]) -> dict[str, Any]:
	return {
		"name": "Attendance Settings",
		"configuration": contract["configuration"],
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}
