"""Dependency-light HCM Time and Attendance lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import (
		ATTENDANCE_EVENT_STREAM,
		STREAMING,
		SUPPORTED_ATTENDANCE_AGENT_ROLES,
		SUPPORTED_ATTENDANCE_AGENT_RUNTIMES,
		SUPPORTED_ENTRY_METHODS,
		SUPPORTED_ENTRY_TYPES,
		SUPPORTED_EXCEPTION_TYPES,
		SUPPORTED_LEAVE_TYPES,
		SUPPORTED_SCHEDULE_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import (  # type: ignore
		ATTENDANCE_EVENT_STREAM,
		STREAMING,
		SUPPORTED_ATTENDANCE_AGENT_ROLES,
		SUPPORTED_ATTENDANCE_AGENT_RUNTIMES,
		SUPPORTED_ENTRY_METHODS,
		SUPPORTED_ENTRY_TYPES,
		SUPPORTED_EXCEPTION_TYPES,
		SUPPORTED_LEAVE_TYPES,
		SUPPORTED_SCHEDULE_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)


class TimeAttendanceError(Exception):
	"""Base exception for attendance operations."""


class TimeAttendanceNotFoundError(TimeAttendanceError):
	"""Raised when an attendance lifecycle record is not found."""


class TimeAttendanceLifecycleService:
	"""In-memory executable service for Time and Attendance lifecycle packets."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None, *_: Any, **__: Any) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.policies: dict[str, dict[str, Any]] = {}
		self.schedules: dict[str, dict[str, Any]] = {}
		self.shifts: dict[str, dict[str, Any]] = {}
		self.time_entries: dict[str, dict[str, Any]] = {}
		self.breaks: dict[str, dict[str, Any]] = {}
		self.timesheets: dict[str, dict[str, Any]] = {}
		self.leave_requests: dict[str, dict[str, Any]] = {}
		self.exceptions: dict[str, dict[str, Any]] = {}
		self.payroll_exports: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"tenant_context_present": True,
			"operation": operation,
			"operation_type": "write",
			"policy_attached": True,
			"audit_enabled": True,
		}

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": ATTENDANCE_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_time_policy(self, policy_id: str, tenant_id: str, name: str, timezone: str, workweek: list[str], overtime_threshold_hours: float = 40.0) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_time_policy")
		context.update({
			"name_present": bool(name),
			"timezone_present": bool(timezone),
			"workweek_present": bool(workweek),
			"overtime_threshold_present": overtime_threshold_hours is not None,
			"overtime_threshold_positive": overtime_threshold_hours is not None and float(overtime_threshold_hours) > 0,
		})
		self._assert_rules(context)
		record = {"id": self._record_id("policy", policy_id), "type": "time_policy", "kind": "policy", "tenant_id": tenant, "name": name, "timezone": timezone, "workweek": list(workweek), "overtime_threshold_hours": float(overtime_threshold_hours), "status": "active", "created_at": self._now()}
		self.policies[record["id"]] = record
		self._emit(tenant, "attendance_policy_created", record)
		return deepcopy(record)

	def create_schedule(self, schedule_id: str, tenant_id: str, employee_id: str, policy_id: str, schedule_type: str, start_date: str, end_date: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		policy = self.policies.get(policy_id)
		context = self._base_context(tenant, "create_schedule")
		context.update({
			"employee_present": bool(employee_id),
			"policy_present": bool(policy and policy["tenant_id"] == tenant),
			"schedule_type_supported": schedule_type in SUPPORTED_SCHEDULE_TYPES,
			"start_date_present": bool(start_date),
			"end_date_present": bool(end_date),
		})
		self._assert_rules(context)
		record = {"id": self._record_id("schedule", schedule_id), "type": "work_schedule", "kind": "schedule", "tenant_id": tenant, "employee_id": employee_id, "policy_id": policy_id, "schedule_type": schedule_type, "start_date": start_date, "end_date": end_date, "status": "active", "created_at": self._now()}
		self.schedules[record["id"]] = record
		self._emit(tenant, "attendance_schedule_created", record)
		return deepcopy(record)

	def create_shift(self, shift_id: str, tenant_id: str, schedule_id: str, shift_date: str, start_time: str, end_time: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		schedule = self.schedules.get(schedule_id)
		context = self._base_context(tenant, "create_shift")
		context.update({
			"schedule_present": bool(schedule and schedule["tenant_id"] == tenant),
			"shift_date_present": bool(shift_date),
			"start_time_present": bool(start_time),
			"end_time_present": bool(end_time),
		})
		self._assert_rules(context)
		record = {"id": self._record_id("shift", shift_id), "type": "attendance_shift", "kind": "shift", "tenant_id": tenant, "schedule_id": schedule_id, "employee_id": schedule["employee_id"], "shift_date": shift_date, "start_time": start_time, "end_time": end_time, "status": "planned", "created_at": self._now()}
		self.shifts[record["id"]] = record
		self._emit(tenant, "attendance_shift_created", record)
		return deepcopy(record)

	def record_time_entry(self, entry_id: str, tenant_id: str, employee_id: str, shift_id: str, entry_type: str, method: str, clock_in: str, clock_out: str | None = None, device_id: str | None = None, geofence_verified: bool = True, biometric_confidence: float | None = None, reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		shift = self.shifts.get(shift_id)
		tracked_method = method in {"mobile", "kiosk", "biometric"}
		low_biometric_confidence = biometric_confidence is not None and biometric_confidence < 0.85
		context = self._base_context(tenant, "record_time_entry")
		context.update({
			"employee_present": bool(employee_id),
			"shift_present": bool(shift and shift["tenant_id"] == tenant),
			"entry_type_supported": entry_type in SUPPORTED_ENTRY_TYPES,
			"entry_method_supported": method in SUPPORTED_ENTRY_METHODS,
			"clock_in_present": bool(clock_in),
			"tracked_method": tracked_method,
			"device_present": bool(device_id),
			"geofence_verified": bool(geofence_verified),
			"biometric_low_confidence": low_biometric_confidence,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		hours = self._calculate_hours(clock_in, clock_out)
		record = {"id": self._record_id("entry", entry_id), "type": "time_entry", "kind": "time_entry", "tenant_id": tenant, "employee_id": employee_id, "shift_id": shift_id, "entry_type": entry_type, "method": method, "clock_in": clock_in, "clock_out": clock_out, "device_id": device_id, "geofence_verified": geofence_verified, "biometric_confidence": biometric_confidence, "reviewed_by": reviewed_by, "hours": hours, "status": "recorded", "created_at": self._now()}
		self.time_entries[record["id"]] = record
		self._emit(tenant, "time_entry_recorded", record)
		return deepcopy(record)

	def record_break(self, break_id: str, tenant_id: str, time_entry_id: str, break_type: str, start_time: str, end_time: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		entry = self.time_entries.get(time_entry_id)
		context = self._base_context(tenant, "record_break")
		context.update({"time_entry_present": bool(entry and entry["tenant_id"] == tenant), "start_time_present": bool(start_time), "end_time_present": bool(end_time)})
		self._assert_rules(context)
		record = {"id": self._record_id("break", break_id), "type": "attendance_break", "kind": "break", "tenant_id": tenant, "time_entry_id": time_entry_id, "break_type": break_type, "start_time": start_time, "end_time": end_time, "status": "recorded", "created_at": self._now()}
		self.breaks[record["id"]] = record
		self._emit(tenant, "break_recorded", record)
		return deepcopy(record)

	def submit_timesheet(self, timesheet_id: str, tenant_id: str, employee_id: str, period_start: str, period_end: str, entry_ids: list[str], submitted_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		entries = [self.time_entries.get(entry_id) for entry_id in entry_ids]
		valid_entries = [entry for entry in entries if entry and entry["tenant_id"] == tenant and entry["employee_id"] == employee_id]
		total_hours = round(sum(float(entry.get("hours") or 0) for entry in valid_entries), 2)
		context = self._base_context(tenant, "submit_timesheet")
		context.update({
			"employee_present": bool(employee_id),
			"period_present": bool(period_start and period_end),
			"entries_present": bool(entry_ids and len(valid_entries) == len(entry_ids)),
			"submitter_present": bool(submitted_by),
			"total_hours_negative": total_hours < 0,
		})
		self._assert_rules(context)
		record = {"id": self._record_id("timesheet", timesheet_id), "type": "attendance_timesheet", "kind": "timesheet", "tenant_id": tenant, "employee_id": employee_id, "period_start": period_start, "period_end": period_end, "entry_ids": list(entry_ids), "total_hours": total_hours, "submitted_by": submitted_by, "approved_by": None, "status": "submitted", "created_at": self._now(), "updated_at": self._now()}
		self.timesheets[record["id"]] = record
		self._emit(tenant, "timesheet_submitted", record)
		return deepcopy(record)

	def approve_timesheet(self, timesheet_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		timesheet = self._get_tenant_record(self.timesheets, timesheet_id, tenant, "timesheet")
		context = self._base_context(tenant, "approve_timesheet")
		context.update({"approver_present": bool(approved_by)})
		self._assert_rules(context)
		timesheet["approved_by"] = approved_by
		timesheet["status"] = "approved"
		timesheet["updated_at"] = self._now()
		self._emit(tenant, "timesheet_approved", timesheet)
		return deepcopy(timesheet)

	def request_leave(self, leave_id: str, tenant_id: str, employee_id: str, leave_type: str, start_date: str, end_date: str, reason: str, approved_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		review_required = leave_type == "unpaid" or self._date_span_days(start_date, end_date) > 10
		context = self._base_context(tenant, "request_leave")
		context.update({
			"employee_present": bool(employee_id),
			"leave_type_supported": leave_type in SUPPORTED_LEAVE_TYPES,
			"start_date_present": bool(start_date),
			"end_date_present": bool(end_date),
			"reason_present": bool(reason),
			"review_required": review_required,
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		record = {"id": self._record_id("leave", leave_id), "type": "attendance_leave_request", "kind": "leave_request", "tenant_id": tenant, "employee_id": employee_id, "leave_type": leave_type, "start_date": start_date, "end_date": end_date, "reason": reason, "approved_by": approved_by, "status": "approved" if approved_by else "requested", "created_at": self._now()}
		self.leave_requests[record["id"]] = record
		self._emit(tenant, "leave_requested", record)
		return deepcopy(record)

	def record_exception(self, exception_id: str, tenant_id: str, employee_id: str, exception_type: str, severity: str, description: str, owner_id: str | None = None, entry_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "record_exception")
		context.update({
			"employee_present": bool(employee_id),
			"exception_type_supported": exception_type in SUPPORTED_EXCEPTION_TYPES,
			"high_severity": severity == "high",
			"owner_present": bool(owner_id),
		})
		self._assert_rules(context)
		record = {"id": self._record_id("exception", exception_id), "type": "attendance_exception", "kind": "exception", "tenant_id": tenant, "employee_id": employee_id, "entry_id": entry_id, "exception_type": exception_type, "severity": severity, "description": description, "owner_id": owner_id, "status": "open", "created_at": self._now()}
		self.exceptions[record["id"]] = record
		self._emit(tenant, "attendance_exception_recorded", record)
		return deepcopy(record)

	def create_payroll_export(self, export_id: str, tenant_id: str, period_start: str, period_end: str, timesheet_ids: list[str], approved_by: str, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		timesheets = [self.timesheets.get(timesheet_id) for timesheet_id in timesheet_ids]
		valid_timesheets = [timesheet for timesheet in timesheets if timesheet and timesheet["tenant_id"] == tenant]
		all_approved = bool(valid_timesheets) and all(timesheet["status"] == "approved" for timesheet in valid_timesheets)
		context = self._base_context(tenant, "create_payroll_export")
		context.update({
			"period_present": bool(period_start and period_end),
			"timesheets_present": bool(timesheet_ids and len(valid_timesheets) == len(timesheet_ids)),
			"all_timesheets_approved": all_approved,
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		if event_stream != "bytewax":
			self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "attendance_batch", "event_stream": "queue"})
		total_hours = round(sum(float(timesheet["total_hours"]) for timesheet in valid_timesheets), 2)
		record = {"id": self._record_id("export", export_id), "type": "attendance_payroll_export", "kind": "payroll_export", "tenant_id": tenant, "period_start": period_start, "period_end": period_end, "timesheet_ids": list(timesheet_ids), "total_hours": total_hours, "approved_by": approved_by, "stream": ATTENDANCE_EVENT_STREAM, "processor": "bytewax", "status": "ready", "created_at": self._now()}
		self.payroll_exports[record["id"]] = record
		self._emit(tenant, "attendance_payroll_export_created", record)
		return deepcopy(record)

	def register_attendance_agent(self, tenant_id: str, name: str, runtime: str, role: str, purpose: str, owner_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_attendance_agent")
		context.update({"runtime_supported": runtime in SUPPORTED_ATTENDANCE_AGENT_RUNTIMES, "role_supported": role in SUPPORTED_ATTENDANCE_AGENT_ROLES})
		self._assert_rules(context)
		record = {"id": self._record_id("agent"), "type": "attendance_agent", "kind": "agent", "tenant_id": tenant, "name": name, "runtime": runtime, "role": role, "purpose": purpose, "owner_id": owner_id, "status": "active", "created_at": self._now()}
		self.agents[record["id"]] = record
		self._emit(tenant, "attendance_agent_registered", record)
		return deepcopy(record)

	def validate_attendance_agent_action(self, tenant_id: str, privileged_action: bool, human_approved: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return evaluate_capability_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "agent_action", "privileged_action": privileged_action, "human_approved": human_approved})

	def validate_batch(self, tenant_id: str, record_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if event_stream != "bytewax":
			self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "attendance_batch", "event_stream": "queue"})
		return {"tenant_id": tenant, "record_count": int(record_count), "processor": "bytewax", "event_stream": ATTENDANCE_EVENT_STREAM, "accepted": True}

	def create_record(self, payload: dict[str, Any]) -> dict[str, Any]:
		tenant = self._tenant(payload.get("tenant_id"))
		record = {"id": self._record_id("record", payload.get("id")), "type": payload.get("type", "attendance_record"), "kind": payload.get("kind", "generic"), "tenant_id": tenant, "status": payload.get("status", "active"), "created_at": self._now(), **payload}
		self._emit(tenant, "attendance_record_created", record)
		return deepcopy(record)

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		def count(records: dict[str, dict[str, Any]]) -> int:
			return sum(1 for record in records.values() if record["tenant_id"] == tenant)
		return {
			"tenant_id": tenant,
			"policy_count": count(self.policies),
			"schedule_count": count(self.schedules),
			"shift_count": count(self.shifts),
			"time_entry_count": count(self.time_entries),
			"timesheet_count": count(self.timesheets),
			"leave_request_count": count(self.leave_requests),
			"exception_count": count(self.exceptions),
			"payroll_export_count": count(self.payroll_exports),
			"agent_count": count(self.agents),
			"audit_event_count": sum(1 for event in self._audit_events if event["tenant_id"] == tenant),
			"streaming": deepcopy(STREAMING),
		}

	def list_records(self, tenant_id: str, record_type: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		stores = [self.policies, self.schedules, self.shifts, self.time_entries, self.breaks, self.timesheets, self.leave_requests, self.exceptions, self.payroll_exports, self.agents]
		records = [record for store in stores for record in store.values() if record["tenant_id"] == tenant]
		if record_type:
			records = [record for record in records if record["type"] == record_type or record["kind"] == record_type]
		return deepcopy(records)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return deepcopy([event for event in self._audit_events if event["tenant_id"] == tenant])

	def _get_tenant_record(self, store: dict[str, dict[str, Any]], record_id: str, tenant_id: str, label: str) -> dict[str, Any]:
		record = store.get(record_id)
		if not record or record["tenant_id"] != tenant_id:
			raise TimeAttendanceNotFoundError(f"{label}_not_found")
		return record

	def _calculate_hours(self, clock_in: str, clock_out: str | None) -> float:
		if not clock_in or not clock_out:
			return 0.0
		try:
			start = datetime.fromisoformat(clock_in.replace("Z", "+00:00"))
			end = datetime.fromisoformat(clock_out.replace("Z", "+00:00"))
			return round(max((end - start).total_seconds() / 3600, 0), 2)
		except ValueError:
			return 0.0

	def _date_span_days(self, start_date: str, end_date: str) -> int:
		try:
			start = datetime.fromisoformat(start_date).date()
			end = datetime.fromisoformat(end_date).date()
			return (end - start).days + 1
		except ValueError:
			return 0


TimeAttendanceService = TimeAttendanceLifecycleService
TimeEntryService = TimeAttendanceLifecycleService
AttendanceScheduleService = TimeAttendanceLifecycleService
AttendanceComplianceService = TimeAttendanceLifecycleService
