"""
Time & Attendance — Flask Blueprint REST API

url_prefix: /hcm/time-attendance/api/v1

All endpoints are async-compatible via asgiref or a sync adapter.
This file also preserves the thin dict-API helpers from the original
capability contract layer for backward compatibility.

Copyright © 2025 Datacraft. Author: Nyimbi Odero
"""
from __future__ import annotations

import asyncio
import functools
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from flask import Blueprint, Response, Request, jsonify, request

from .context import resolve_current_user_context

logger = logging.getLogger(__name__)


def get_current_user_context(request: Request) -> dict:
	"""Resolve current user context from the active request."""
	return resolve_current_user_context(request)

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

bp = Blueprint(
	"tat_api",
	__name__,
	url_prefix="/hcm/time-attendance/api/v1",
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
				future = pool.submit(asyncio.run, coro)
				return future.result(timeout=30)
		return asyncio.run(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _svc() -> Any:
	"""
	Build a TimeAttendanceService from the current request context.
	Expects headers: X-Tenant-Id, X-Actor-Id (or falls back to request.args).
	DB session is a stub; replace with real session factory.
	"""
	try:
		from .service import TimeAttendanceService
	except ImportError:
		from service import TimeAttendanceService  # type: ignore[no-redef]

	tenant_id = (
		request.headers.get("X-Tenant-Id")
		or request.args.get("tenant_id")
		or "default"
	)
	actor_id = (
		request.headers.get("X-Actor-Id")
		or request.args.get("actor_id")
		or "system"
	)

	# Lazy import stub — replace with real async DB session in deployment
	try:
		from .database import get_db_session
		db = get_db_session()
	except Exception:
		db = _StubDB()

	return TimeAttendanceService(db, tenant_id, actor_id)


class _StubDB:
	"""No-op stub so the Blueprint loads without a live DB."""

	async def execute(self, *a: Any, **kw: Any) -> None:
		raise RuntimeError("No database session configured")

	async def fetch(self, *a: Any, **kw: Any) -> list:
		raise RuntimeError("No database session configured")

	async def fetchrow(self, *a: Any, **kw: Any) -> None:
		raise RuntimeError("No database session configured")


def _ok(data: Any, status: int = 200) -> Response:
	return jsonify({"ok": True, "data": data}), status


def _err(message: str, status: int = 400) -> Response:
	return jsonify({"ok": False, "error": message}), status


def _handle(coro: Any) -> Response:
	"""Execute async service call, return structured JSON response."""
	try:
		result = _run(coro)
		return _ok(result)
	except PermissionError as exc:
		return _err(str(exc), 403)
	except KeyError as exc:
		return _err(f"Not found: {exc}", 404)
	except ValueError as exc:
		return _err(str(exc), 422)
	except Exception as exc:
		logger.exception("TAT API error: %s", exc)
		return _err(str(exc), 500)


def _body() -> dict[str, Any]:
	return request.get_json(force=True, silent=True) or {}


def _date(val: str | None) -> date | None:
	if not val:
		return None
	return datetime.strptime(val, "%Y-%m-%d").date()


def _int(val: str | None, default: int = 50) -> int:
	try:
		return int(val) if val else default
	except (ValueError, TypeError):
		return default


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@bp.get("/health")
def health() -> Response:
	return _ok({"status": "ok", "capability": "tat_time_attendance"})


# ---------------------------------------------------------------------------
# Time Policies
# ---------------------------------------------------------------------------

@bp.get("/policies")
def list_policies() -> Response:
	svc = _svc()
	return _handle(svc.list_time_policies(
		limit=_int(request.args.get("limit"), 50),
		offset=_int(request.args.get("offset"), 0),
	))


@bp.post("/policies")
def create_policy() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.create_time_policy(
		name=body.get("name", ""),
		timezone=body.get("timezone", "UTC"),
		workweek=body.get("workweek", ["Mon", "Tue", "Wed", "Thu", "Fri"]),
		overtime_threshold_daily=float(body.get("overtime_threshold_daily", 8.0)),
		overtime_threshold_weekly=float(body.get("overtime_threshold_weekly", 40.0)),
		double_time_threshold_daily=float(body.get("double_time_threshold_daily", 12.0)),
		overtime_multiplier=float(body.get("overtime_multiplier", 1.5)),
		holiday_pay_multiplier=float(body.get("holiday_pay_multiplier", 2.0)),
		min_rest_between_shifts_h=float(body.get("min_rest_between_shifts_h", 11.0)),
		max_consecutive_days=int(body.get("max_consecutive_days", 6)),
		max_weekly_hours=float(body.get("max_weekly_hours", 48.0)),
		break_rules=body.get("break_rules"),
		toil_enabled=bool(body.get("toil_enabled", False)),
		comp_time_enabled=bool(body.get("comp_time_enabled", False)),
		annualised_hours_enabled=bool(body.get("annualised_hours_enabled", False)),
		contracted_annual_hours=body.get("contracted_annual_hours"),
		medical_cert_threshold_days=int(body.get("medical_cert_threshold_days", 3)),
		metadata=body.get("metadata"),
	))


@bp.get("/policies/<policy_id>")
def get_policy(policy_id: str) -> Response:
	return _handle(_svc().get_time_policy(policy_id))


@bp.put("/policies/<policy_id>")
def update_policy(policy_id: str) -> Response:
	body = _body()
	return _handle(_svc().update_time_policy(policy_id, **body))


@bp.delete("/policies/<policy_id>")
def delete_policy(policy_id: str) -> Response:
	async def _do() -> dict[str, Any]:
		await _svc().delete_time_policy(policy_id)
		return {"deleted": True, "id": policy_id}
	return _handle(_do())


# ---------------------------------------------------------------------------
# Shift Schedules
# ---------------------------------------------------------------------------

@bp.get("/schedules")
def list_schedules() -> Response:
	return _handle(_svc().list_shift_schedules(
		department_id=request.args.get("department_id"),
		limit=_int(request.args.get("limit"), 50),
		offset=_int(request.args.get("offset"), 0),
	))


@bp.post("/schedules")
def create_schedule() -> Response:
	body = _body()
	eff = _date(body.get("effective_date")) or date.today()
	end = _date(body.get("end_date"))
	svc = _svc()
	return _handle(svc.create_shift_schedule(
		policy_id=body.get("policy_id", ""),
		schedule_name=body.get("schedule_name", ""),
		schedule_type=body.get("schedule_type", "fixed"),
		effective_date=eff,
		patterns=body.get("patterns", []),
		end_date=end,
		department_id=body.get("department_id"),
		location_id=body.get("location_id"),
		description=body.get("description"),
		allow_overtime=bool(body.get("allow_overtime", True)),
		allow_shift_swapping=bool(body.get("allow_shift_swapping", True)),
		metadata=body.get("metadata"),
	))


@bp.get("/schedules/<schedule_id>")
def get_schedule(schedule_id: str) -> Response:
	return _handle(_svc().get_shift_schedule(schedule_id))


# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------

@bp.get("/shifts")
def list_shifts() -> Response:
	svc = _svc()
	return _handle(svc.list_shifts(
		employee_id=request.args.get("employee_id"),
		from_date=_date(request.args.get("from_date")),
		to_date=_date(request.args.get("to_date")),
		limit=_int(request.args.get("limit"), 100),
		offset=_int(request.args.get("offset"), 0),
	))


@bp.post("/shifts")
def create_shift() -> Response:
	body = _body()
	svc = _svc()
	async def _do() -> Any:
		ps = body.get("planned_start") or ""
		pe = body.get("planned_end") or ""
		if not ps or not pe:
			raise ValueError("planned_start and planned_end are required")
		planned_start = datetime.fromisoformat(ps)
		planned_end = datetime.fromisoformat(pe)
		return await svc.create_shift(
			schedule_id=body.get("schedule_id", ""),
			employee_id=body.get("employee_id", ""),
			shift_date=_date(body.get("shift_date")) or planned_start.date(),
			planned_start=planned_start,
			planned_end=planned_end,
			location_id=body.get("location_id"),
			notes=body.get("notes"),
		)
	return _handle(_do())


@bp.get("/shifts/<shift_id>")
def get_shift(shift_id: str) -> Response:
	return _handle(_svc().get_shift(shift_id))


# ---------------------------------------------------------------------------
# Time Entries (Clock-in / Clock-out)
# ---------------------------------------------------------------------------

@bp.get("/time-entries")
def list_entries() -> Response:
	svc = _svc()
	return _handle(svc.list_time_entries(
		employee_id=request.args.get("employee_id"),
		from_date=_date(request.args.get("from_date")),
		to_date=_date(request.args.get("to_date")),
		status=request.args.get("status"),
		limit=_int(request.args.get("limit"), 100),
		offset=_int(request.args.get("offset"), 0),
	))


@bp.post("/time-entries/clock-in")
def clock_in() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.clock_in(
		employee_id=body.get("employee_id", ""),
		shift_id=body.get("shift_id", ""),
		entry_type=body.get("entry_type", "regular"),
		method=body.get("method", "web"),
		device_id=body.get("device_id"),
		latitude=body.get("latitude"),
		longitude=body.get("longitude"),
		biometric_confidence=body.get("biometric_confidence"),
		ip_address=request.remote_addr,
		cost_center=body.get("cost_center"),
		notes=body.get("notes"),
	))


@bp.post("/time-entries/<entry_id>/clock-out")
def clock_out(entry_id: str) -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.clock_out(
		entry_id=entry_id,
		latitude=body.get("latitude"),
		longitude=body.get("longitude"),
		notes=body.get("notes"),
	))


@bp.get("/time-entries/<entry_id>")
def get_entry(entry_id: str) -> Response:
	return _handle(_svc().get_time_entry(entry_id))


@bp.put("/time-entries/<entry_id>")
def update_entry(entry_id: str) -> Response:
	body = _body()
	return _handle(_svc().update_time_entry(entry_id, **body))


@bp.delete("/time-entries/<entry_id>")
def delete_entry(entry_id: str) -> Response:
	async def _do() -> dict[str, Any]:
		await _svc().delete_time_entry(entry_id)
		return {"deleted": True, "id": entry_id}
	return _handle(_do())


# ---------------------------------------------------------------------------
# Breaks
# ---------------------------------------------------------------------------

@bp.post("/time-entries/<entry_id>/breaks")
def record_break(entry_id: str) -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.record_break(
		time_entry_id=entry_id,
		break_type=body.get("break_type", "meal"),
		break_start=datetime.fromisoformat(body.get("break_start", "")),
		break_end=datetime.fromisoformat(body.get("break_end", "")),
		is_paid=bool(body.get("is_paid", False)),
	))


# ---------------------------------------------------------------------------
# Timesheets
# ---------------------------------------------------------------------------

@bp.get("/timesheets")
def list_timesheets() -> Response:
	svc = _svc()
	return _handle(svc.list_timesheets(
		employee_id=request.args.get("employee_id"),
		status=request.args.get("status"),
		from_date=_date(request.args.get("from_date")),
		to_date=_date(request.args.get("to_date")),
		limit=_int(request.args.get("limit"), 50),
		offset=_int(request.args.get("offset"), 0),
	))


@bp.post("/timesheets/process")
def process_timesheet() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.process_timesheet(
		employee_id=body.get("employee_id", ""),
		period_start=_date(body.get("period_start")) or date.today(),
		period_end=_date(body.get("period_end")) or date.today(),
		hourly_rate=Decimal(str(body["hourly_rate"])) if body.get("hourly_rate") else None,
		currency=body.get("currency", "USD"),
	))


@bp.get("/timesheets/<timesheet_id>")
def get_timesheet(timesheet_id: str) -> Response:
	return _handle(_svc().get_timesheet(timesheet_id))


@bp.post("/timesheets/<timesheet_id>/submit")
def submit_timesheet(timesheet_id: str) -> Response:
	return _handle(_svc().submit_timesheet(timesheet_id))


@bp.post("/timesheets/<timesheet_id>/approve")
def approve_timesheet(timesheet_id: str) -> Response:
	return _handle(_svc().approve_timesheet(timesheet_id))


@bp.post("/timesheets/<timesheet_id>/reject")
def reject_timesheet(timesheet_id: str) -> Response:
	body = _body()
	return _handle(_svc().reject_timesheet(timesheet_id, reason=body.get("reason", "")))


# ---------------------------------------------------------------------------
# Overtime
# ---------------------------------------------------------------------------

@bp.get("/overtime")
def calculate_overtime() -> Response:
	svc = _svc()
	return _handle(svc.calculate_overtime(
		employee_id=request.args.get("employee_id", ""),
		period_start=_date(request.args.get("from_date")) or date.today(),
		period_end=_date(request.args.get("to_date")) or date.today(),
		policy_id=request.args.get("policy_id", ""),
	))


@bp.post("/overtime/requests")
def request_overtime() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.request_overtime(
		employee_id=body.get("employee_id", ""),
		shift_id=body.get("shift_id", ""),
		requested_hours=float(body.get("requested_hours", 0)),
		reason=body.get("reason", ""),
	))


@bp.post("/overtime/requests/<request_id>/approve")
def approve_overtime(request_id: str) -> Response:
	return _handle(_svc().approve_overtime_request(request_id))


@bp.post("/overtime/requests/<request_id>/reject")
def reject_overtime(request_id: str) -> Response:
	body = _body()
	return _handle(_svc().reject_overtime_request(request_id, reason=body.get("reason", "")))


# ---------------------------------------------------------------------------
# Leave
# ---------------------------------------------------------------------------

@bp.get("/leave")
def list_leave() -> Response:
	svc = _svc()
	return _handle(svc.list_leave_requests(
		employee_id=request.args.get("employee_id"),
		leave_type=request.args.get("leave_type"),
		status=request.args.get("status"),
		from_date=_date(request.args.get("from_date")),
		to_date=_date(request.args.get("to_date")),
		limit=_int(request.args.get("limit"), 50),
		offset=_int(request.args.get("offset"), 0),
	))


@bp.post("/leave")
def create_leave() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.request_leave(
		employee_id=body.get("employee_id", ""),
		leave_type=body.get("leave_type", "vacation"),
		start_date=_date(body.get("start_date")) or date.today(),
		end_date=_date(body.get("end_date")) or date.today(),
		reason=body.get("reason"),
		is_emergency=bool(body.get("is_emergency", False)),
		is_half_day=bool(body.get("is_half_day", False)),
		half_day_portion=body.get("half_day_portion"),
		is_statutory=bool(body.get("is_statutory", False)),
		statutory_type=body.get("statutory_type"),
		statutory_jurisdiction=body.get("statutory_jurisdiction"),
		medical_cert_attached=bool(body.get("medical_cert_attached", False)),
		attachments=body.get("attachments"),
		metadata=body.get("metadata"),
	))


@bp.get("/leave/<request_id>")
def get_leave(request_id: str) -> Response:
	return _handle(_svc().get_leave_request(request_id))


@bp.post("/leave/<request_id>/approve")
def approve_leave(request_id: str) -> Response:
	return _handle(_svc().approve_leave_request(request_id))


@bp.post("/leave/<request_id>/reject")
def reject_leave(request_id: str) -> Response:
	body = _body()
	return _handle(_svc().reject_leave_request(request_id, reason=body.get("reason", "")))


@bp.post("/leave/<request_id>/cancel")
def cancel_leave(request_id: str) -> Response:
	return _handle(_svc().cancel_leave_request(request_id))


@bp.get("/leave/entitlement/<employee_id>")
def get_entitlement(employee_id: str) -> Response:
	svc = _svc()
	leave_type = request.args.get("leave_type", "vacation")
	year = _int(request.args.get("year"), date.today().year)
	fte = float(request.args.get("fte", 1.0))
	return _handle(svc.calculate_leave_entitlement_for(employee_id, leave_type, year, fte))


# ---------------------------------------------------------------------------
# Roster generation
# ---------------------------------------------------------------------------

@bp.post("/rosters/generate")
def generate_roster() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.roster_generation(
		schedule_id=body.get("schedule_id", ""),
		period_start=_date(body.get("period_start")) or date.today(),
		period_end=_date(body.get("period_end")) or date.today(),
		employee_ids=body.get("employee_ids", []),
		constraints=body.get("constraints"),
	))


# ---------------------------------------------------------------------------
# Shift swaps
# ---------------------------------------------------------------------------

@bp.post("/shifts/swap")
def request_swap() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.shift_swap_request(
		requester_shift_id=body.get("requester_shift_id", ""),
		target_shift_id=body.get("target_shift_id"),
		target_id=body.get("target_id"),
		reason=body.get("reason"),
	))


@bp.post("/shifts/swap/<swap_id>/approve")
def approve_swap(swap_id: str) -> Response:
	return _handle(_svc().approve_shift_swap(swap_id))


# ---------------------------------------------------------------------------
# Geofence
# ---------------------------------------------------------------------------

@bp.get("/geofences")
def list_geofences() -> Response:
	return _handle(_svc().list_geofence_locations(
		limit=_int(request.args.get("limit"), 50),
		offset=_int(request.args.get("offset"), 0),
	))


@bp.post("/geofences")
def create_geofence() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.create_geofence_location(
		name=body.get("name", ""),
		latitude=float(body.get("latitude", 0)),
		longitude=float(body.get("longitude", 0)),
		radius_metres=float(body.get("radius_metres", 200.0)),
		timezone=body.get("timezone", "UTC"),
		address=body.get("address"),
	))


@bp.post("/geofences/validate")
def validate_geofence() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.gps_geofence_validation(
		employee_id=body.get("employee_id", ""),
		latitude=float(body.get("latitude", 0)),
		longitude=float(body.get("longitude", 0)),
		location_id=body.get("location_id", ""),
	))


# ---------------------------------------------------------------------------
# Biometric sync
# ---------------------------------------------------------------------------

@bp.post("/devices/<device_id>/sync")
def biometric_sync(device_id: str) -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.biometric_device_sync(
		device_id=device_id,
		raw_records=body.get("records", []),
	))


# ---------------------------------------------------------------------------
# Bulk import
# ---------------------------------------------------------------------------

@bp.post("/timesheets/bulk-import")
def bulk_import() -> Response:
	svc = _svc()
	if "file" in request.files:
		csv_content = request.files["file"].read().decode("utf-8")
	else:
		body = _body()
		csv_content = body.get("csv_content", "")
	return _handle(svc.bulk_timesheet_import(csv_content))


# ---------------------------------------------------------------------------
# Flexitime
# ---------------------------------------------------------------------------

@bp.get("/flexitime/<employee_id>")
def flexitime(employee_id: str) -> Response:
	svc = _svc()
	return _handle(svc.flexitime_calculation(
		employee_id=employee_id,
		from_date=_date(request.args.get("from_date")) or date.today().replace(day=1),
		to_date=_date(request.args.get("to_date")) or date.today(),
		policy_id=request.args.get("policy_id", ""),
	))


# ---------------------------------------------------------------------------
# Annualised hours
# ---------------------------------------------------------------------------

@bp.get("/annualised-hours/<employee_id>")
def annualised_hours(employee_id: str) -> Response:
	svc = _svc()
	return _handle(svc.annualised_hours_reconciliation(
		employee_id=employee_id,
		policy_id=request.args.get("policy_id", ""),
		as_of_date=_date(request.args.get("as_of_date")),
	))


# ---------------------------------------------------------------------------
# Comp time
# ---------------------------------------------------------------------------

@bp.post("/comp-time/earn")
def earn_comp_time() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.earn_comp_time(
		employee_id=body.get("employee_id", ""),
		hours=Decimal(str(body.get("hours", 0))),
		time_entry_id=body.get("time_entry_id"),
		reason=body.get("reason"),
		expiry_date=_date(body.get("expiry_date")),
	))


@bp.post("/comp-time/use")
def use_comp_time() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.use_comp_time(
		employee_id=body.get("employee_id", ""),
		hours=Decimal(str(body.get("hours", 0))),
		reason=body.get("reason"),
	))


# ---------------------------------------------------------------------------
# Public holidays
# ---------------------------------------------------------------------------

@bp.get("/public-holidays")
def list_holidays() -> Response:
	svc = _svc()
	return _handle(svc.list_public_holidays(
		jurisdiction=request.args.get("jurisdiction"),
		from_date=_date(request.args.get("from_date")),
		to_date=_date(request.args.get("to_date")),
	))


@bp.post("/public-holidays")
def create_holiday() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.create_public_holiday(
		name=body.get("name", ""),
		holiday_date=_date(body.get("holiday_date")) or date.today(),
		jurisdiction=body.get("jurisdiction", "global"),
		is_statutory=bool(body.get("is_statutory", True)),
		timezone=body.get("timezone", "UTC"),
		substitute_date=_date(body.get("substitute_date")),
	))


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

@bp.get("/exceptions")
def list_exceptions() -> Response:
	svc = _svc()
	return _handle(svc.list_exceptions(
		employee_id=request.args.get("employee_id"),
		status=request.args.get("status"),
		severity=request.args.get("severity"),
		limit=_int(request.args.get("limit"), 50),
		offset=_int(request.args.get("offset"), 0),
	))


@bp.post("/exceptions")
def create_exception() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.record_exception(
		employee_id=body.get("employee_id", ""),
		exception_type=body.get("exception_type", "late_arrival"),
		severity=body.get("severity", "medium"),
		description=body.get("description", ""),
		time_entry_id=body.get("time_entry_id"),
		owner_id=body.get("owner_id"),
	))


@bp.post("/exceptions/<exception_id>/resolve")
def resolve_exception(exception_id: str) -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.resolve_exception(exception_id, resolution_notes=body.get("resolution_notes", "")))


# ---------------------------------------------------------------------------
# Payroll export
# ---------------------------------------------------------------------------

@bp.get("/payroll-exports")
def list_exports() -> Response:
	return _handle(_svc().list_payroll_exports(
		limit=_int(request.args.get("limit"), 20),
		offset=_int(request.args.get("offset"), 0),
	))


@bp.post("/payroll-exports")
def create_export() -> Response:
	body = _body()
	svc = _svc()
	return _handle(svc.create_payroll_export(
		period_start=_date(body.get("period_start")) or date.today(),
		period_end=_date(body.get("period_end")) or date.today(),
		timesheet_ids=body.get("timesheet_ids", []),
		notes=body.get("notes"),
	))


@bp.get("/payroll-exports/<export_id>")
def get_export(export_id: str) -> Response:
	return _handle(_svc().get_payroll_export(export_id))


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@bp.get("/reports/<report_type>")
def report(report_type: str) -> Response:
	svc = _svc()
	return _handle(svc.generate_attendance_report(
		report_type=report_type,
		from_date=_date(request.args.get("from_date")) or date.today().replace(day=1),
		to_date=_date(request.args.get("to_date")) or date.today(),
		employee_id=request.args.get("employee_id"),
		department_id=request.args.get("department_id"),
	))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@bp.get("/dashboard")
def dashboard() -> Response:
	return _handle(_svc().dashboard_summary())


# ---------------------------------------------------------------------------
# Backward-compatible thin-wrapper helpers (from original capability contract)
# These must work when api.py is loaded directly via importlib (no package).
# ---------------------------------------------------------------------------

try:
	from .lifecycle import TimeAttendanceLifecycleService as _LegacySvc
except ImportError:  # direct-load via importlib (no package context)
	from lifecycle import TimeAttendanceLifecycleService as _LegacySvc  # type: ignore[no-redef]

_LEGACY = _LegacySvc()


def service() -> Any:
	"""Return the process-local in-memory lifecycle service (legacy API)."""
	return _LEGACY


def create_time_policy(payload: dict[str, Any]) -> dict[str, Any]:
	"""Legacy dict-API wrapper: create a time policy."""
	return _LEGACY.create_time_policy(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("name", ""),
		payload.get("timezone", "UTC"),
		payload.get("workweek", []),
		payload.get("overtime_threshold_hours", 40.0),
	)


def create_schedule(payload: dict[str, Any]) -> dict[str, Any]:
	return _LEGACY.create_schedule(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("employee_id", ""),
		payload.get("policy_id", ""),
		payload.get("schedule_type", "fixed"),
		payload.get("start_date", ""),
		payload.get("end_date", ""),
	)


def record_time_entry(payload: dict[str, Any]) -> dict[str, Any]:
	return _LEGACY.record_time_entry(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("employee_id", ""),
		payload.get("shift_id", ""),
		payload.get("entry_type", "regular"),
		payload.get("method", "web"),
		payload.get("clock_in", ""),
		payload.get("clock_out"),
		payload.get("device_id"),
		payload.get("geofence_verified", True),
		payload.get("biometric_confidence"),
		payload.get("reviewed_by"),
	)


def submit_timesheet(payload: dict[str, Any]) -> dict[str, Any]:
	return _LEGACY.submit_timesheet(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("employee_id", ""),
		payload.get("period_start", ""),
		payload.get("period_end", ""),
		payload.get("entry_ids", []),
		payload.get("submitted_by", ""),
	)


def approve_timesheet(payload: dict[str, Any]) -> dict[str, Any]:
	return _LEGACY.approve_timesheet(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("approved_by", ""),
	)


def request_leave(payload: dict[str, Any]) -> dict[str, Any]:
	return _LEGACY.request_leave(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("employee_id", ""),
		payload.get("leave_type", "vacation"),
		payload.get("start_date", ""),
		payload.get("end_date", ""),
		payload.get("reason", ""),
		payload.get("approved_by"),
	)


def record_exception(payload: dict[str, Any]) -> dict[str, Any]:
	return _LEGACY.record_exception(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("employee_id", ""),
		payload.get("exception_type", "late_arrival"),
		payload.get("severity", "medium"),
		payload.get("description", ""),
		payload.get("owner_id"),
		payload.get("entry_id"),
	)


def create_payroll_export(payload: dict[str, Any]) -> dict[str, Any]:
	return _LEGACY.create_payroll_export(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("period_start", ""),
		payload.get("period_end", ""),
		payload.get("timesheet_ids", []),
		payload.get("approved_by", ""),
		payload.get("event_stream", "bytewax"),
	)


def register_attendance_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _LEGACY.register_attendance_agent(
		payload.get("tenant_id", "default"),
		payload.get("name", "Attendance Agent"),
		payload.get("runtime", "codex"),
		payload.get("role", "attendance_reviewer"),
		payload.get("purpose", "review attendance events"),
		payload.get("owner_id"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return _LEGACY.create_record(payload)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return _LEGACY.dashboard_summary(tenant_id)


def audit_events(tenant_id: str = "default") -> list[dict[str, Any]]:
	return _LEGACY.audit_events(tenant_id)

def create_app(config=None):
    """Create standalone Flask application for Time & Attendance capability."""
    try:
        from flask import Flask
        app = Flask(__name__)
        if config: app.config.update(config)
        try:
            from .app import create_app as _ca
            return _ca(config)
        except Exception:
            pass
        @app.get("/health")
        def _health():
            from flask import jsonify
            return jsonify({"status": "ok", "capability": "tat_time_attendance"})
        return app
    except ImportError:
        return None

# Backward-compatibility alias
get_current_user = get_current_user_context

def get_service():
    """Return the default capability service instance."""
    try:
        from .service import TimeAndAttendanceService
        return TimeAndAttendanceService(tenant_id="default")
    except Exception:
        return None
