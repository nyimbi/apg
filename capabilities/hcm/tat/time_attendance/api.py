"""Dependency-light API helpers for HCM Time and Attendance."""

from __future__ import annotations

from typing import Any

try:
	from .service import TimeAttendanceLifecycleService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import TimeAttendanceLifecycleService  # type: ignore


_SERVICE = TimeAttendanceLifecycleService()


def service() -> TimeAttendanceLifecycleService:
	"""Return the process-local Time and Attendance lifecycle service."""
	return _SERVICE


def create_time_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_time_policy(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("name", ""),
		payload.get("timezone", ""),
		payload.get("workweek", []),
		payload.get("overtime_threshold_hours", 40.0),
	)


def create_schedule(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_schedule(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("employee_id", ""),
		payload.get("policy_id", ""),
		payload.get("schedule_type", "fixed"),
		payload.get("start_date", ""),
		payload.get("end_date", ""),
	)


def create_shift(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_shift(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("schedule_id", ""),
		payload.get("shift_date", ""),
		payload.get("start_time", ""),
		payload.get("end_time", ""),
	)


def record_time_entry(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_time_entry(
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


def record_break(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_break(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("time_entry_id", ""),
		payload.get("break_type", "meal"),
		payload.get("start_time", ""),
		payload.get("end_time", ""),
	)


def submit_timesheet(payload: dict[str, Any]) -> dict[str, Any]:
	return service().submit_timesheet(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("employee_id", ""),
		payload.get("period_start", ""),
		payload.get("period_end", ""),
		payload.get("entry_ids", []),
		payload.get("submitted_by", ""),
	)


def approve_timesheet(payload: dict[str, Any]) -> dict[str, Any]:
	return service().approve_timesheet(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("approved_by", ""),
	)


def request_leave(payload: dict[str, Any]) -> dict[str, Any]:
	return service().request_leave(
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
	return service().record_exception(
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
	return service().create_payroll_export(
		payload.get("id", ""),
		payload.get("tenant_id", "default"),
		payload.get("period_start", ""),
		payload.get("period_end", ""),
		payload.get("timesheet_ids", []),
		payload.get("approved_by", ""),
		payload.get("event_stream", "bytewax"),
	)


def register_attendance_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return service().register_attendance_agent(
		payload.get("tenant_id", "default"),
		payload.get("name", "Attendance Agent"),
		payload.get("runtime", "codex"),
		payload.get("role", "attendance_reviewer"),
		payload.get("purpose", "review attendance events"),
		payload.get("owner_id"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_record(payload)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return service().dashboard_summary(tenant_id)


def audit_events(tenant_id: str = "default") -> list[dict[str, Any]]:
	return service().audit_events(tenant_id)
