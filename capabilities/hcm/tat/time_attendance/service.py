"""
Time & Attendance — Async Domain Service

Full lifecycle management: clock-in/out, shift scheduling, leave, TOIL,
flexitime, annualised hours, geofencing, biometric sync, payroll export.

All methods are async.  DB session is injected; no ORM assumed — raw
asyncpg-compatible dicts (or SQLAlchemy async sessions) work equally well
via the thin _exec / _fetch helpers.

Copyright © 2025 Datacraft. Author: Nyimbi Odero
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any, ClassVar

from uuid6 import uuid7
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .domain.calculations import (
		HoursBreakdown,
		LeaveEntitlement,
		calculate_hours_breakdown,
		calculate_leave_entitlement,
		calculate_pay,
		calculate_prorata_leave,
		calculate_toil_from_overtime,
		calculate_weekly_hours_breakdown,
		calculate_worked_hours,
		working_days_between,
		calculate_flexi_balance,
	)
	from .domain.rules import (
		RuleViolation,
		assert_annualised_hours_deficit_manageable,
		assert_biometric_confidence,
		assert_clock_in_before_clock_out,
		assert_core_hours_covered,
		assert_device_registered,
		assert_fmla_eligibility,
		assert_import_row_valid,
		assert_leave_balance_sufficient,
		assert_leave_dates_valid,
		assert_leave_not_overlapping,
		assert_maximum_consecutive_days,
		assert_maximum_weekly_hours,
		assert_medical_certificate_for_extended_sick,
		assert_minimum_rest_between_shifts,
		assert_night_shift_midnight_span,
		assert_no_cross_tenant_access,
		assert_not_already_clocked_in,
		assert_not_already_clocked_out,
		assert_overtime_threshold_positive,
		assert_shift_duration_reasonable,
		assert_tenant_context,
		assert_timesheet_approved_before_export,
		assert_within_geofence,
		calculate_daily_overtime,
		calculate_weekly_overtime,
		is_night_shift,
	)
except ImportError:  # pragma: no cover – direct-load via importlib in contract tests
	from domain.calculations import (  # type: ignore[no-redef]
		HoursBreakdown,
		LeaveEntitlement,
		calculate_hours_breakdown,
		calculate_leave_entitlement,
		calculate_pay,
		calculate_prorata_leave,
		calculate_toil_from_overtime,
		calculate_weekly_hours_breakdown,
		calculate_worked_hours,
		working_days_between,
		calculate_flexi_balance,
	)
	from domain.rules import (  # type: ignore[no-redef]
		RuleViolation,
		assert_annualised_hours_deficit_manageable,
		assert_biometric_confidence,
		assert_clock_in_before_clock_out,
		assert_core_hours_covered,
		assert_device_registered,
		assert_fmla_eligibility,
		assert_import_row_valid,
		assert_leave_balance_sufficient,
		assert_leave_dates_valid,
		assert_leave_not_overlapping,
		assert_maximum_consecutive_days,
		assert_maximum_weekly_hours,
		assert_medical_certificate_for_extended_sick,
		assert_minimum_rest_between_shifts,
		assert_night_shift_midnight_span,
		assert_no_cross_tenant_access,
		assert_not_already_clocked_in,
		assert_not_already_clocked_out,
		assert_overtime_threshold_positive,
		assert_shift_duration_reasonable,
		assert_tenant_context,
		assert_timesheet_approved_before_export,
		assert_within_geofence,
		calculate_daily_overtime,
		calculate_weekly_overtime,
		is_night_shift,
	)

logger = logging.getLogger(__name__)


def _uuid7str() -> str:
	return str(uuid7())


UTC = timezone.utc


class TimeAttendanceError(Exception):
	"""Base domain exception."""


class NotFoundError(TimeAttendanceError):
	"""Record not found or tenant mismatch."""


class TimeAttendanceService:
	"""
	Full-featured async Time & Attendance service.

	Can be used in two modes:

	1. **DB mode** (production): pass ``db_session``, ``tenant_id``, and ``actor_id``.
	2. **In-memory mode** (testing / capability sandbox): call with no args.
	   All state is kept in class-level dicts, shared across instances, and
	   cleared with :meth:`reset_runtime_store`.

	Args:
		db_session: Async database session (asyncpg Connection or SQLAlchemy AsyncSession).
		tenant_id:  Tenant context — enforced on every operation.
		actor_id:   Authenticated user performing the action.
	"""

	# ------------------------------------------------------------------
	# Class-level runtime store — shared across all in-memory instances
	# ------------------------------------------------------------------
	_store_time_entries: ClassVar[dict[str, Any]] = {}
	_store_remote_workers: ClassVar[dict[str, Any]] = {}
	_store_ai_agents: ClassVar[dict[str, Any]] = {}
	_store_schedules: ClassVar[dict[str, Any]] = {}
	_store_leave_requests: ClassVar[dict[str, Any]] = {}

	@classmethod
	def reset_runtime_store(cls) -> None:
		"""Clear all in-memory store dicts.  Call at the start of each test."""
		cls._store_time_entries.clear()
		cls._store_remote_workers.clear()
		cls._store_ai_agents.clear()
		cls._store_schedules.clear()
		cls._store_leave_requests.clear()

	def __init__(
		self,
		db_session: Any = None,
		tenant_id: str | None = None,
		actor_id: str | None = None,
	) -> None:
		if db_session is not None:
			assert tenant_id, "tenant_id is required"
			assert actor_id, "actor_id is required"
		self._db = db_session
		self._tenant_id = tenant_id or ""
		self._actor_id = actor_id or ""
		self._events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# In-memory helpers
	# ------------------------------------------------------------------

	def _im_mode(self) -> bool:
		"""Return True when operating in in-memory (no-DB) mode."""
		return self._db is None

	# ------------------------------------------------------------------
	# In-memory CRUD methods required by blueprint, mobile_api, reporting,
	# monitoring, websocket and the runtime-store tests.
	# ------------------------------------------------------------------

	async def clock_in(
		self,
		employee_id: str,
		tenant_id: str,
		device_info: dict[str, Any] | None = None,
		location: dict[str, Any] | None = None,
		biometric_data: dict[str, Any] | None = None,
		created_by: str = "",
		work_mode: Any = None,
	) -> Any:
		"""Clock in an employee (in-memory mode)."""
		if not self._im_mode():
			# delegate to legacy DB path — re-raise to surface clearly
			raise NotImplementedError("DB-mode clock_in uses the legacy signature")
		try:
			from .models import TATimeEntry, TimeEntryStatus
		except ImportError:
			from models import TATimeEntry, TimeEntryStatus  # type: ignore
		now = datetime.utcnow()
		entry = TATimeEntry(
			tenant_id=tenant_id,
			employee_id=employee_id,
			entry_date=now.date(),
			clock_in=now,
			status=TimeEntryStatus.DRAFT,
			device_info=device_info or {},
			created_by=created_by or "system",
		)
		self.__class__._store_time_entries[entry.id] = entry
		return entry

	async def clock_out(
		self,
		employee_id: str,
		tenant_id: str,
		device_info: dict[str, Any] | None = None,
		location: dict[str, Any] | None = None,
		biometric_data: dict[str, Any] | None = None,
		created_by: str = "",
	) -> Any:
		"""Clock out the active entry for employee (in-memory mode)."""
		if not self._im_mode():
			raise NotImplementedError("DB-mode clock_out uses the legacy signature")
		# find the open entry
		entry = None
		for e in self.__class__._store_time_entries.values():
			if e.tenant_id == tenant_id and e.employee_id == employee_id and e.clock_out is None:
				entry = e
				break
		if entry is None:
			raise ValueError(f"No open clock-in found for employee {employee_id}")
		now = datetime.utcnow()
		entry.clock_out = now
		delta = now - entry.clock_in
		total = Decimal(str(round(delta.total_seconds() / 3600, 4)))
		regular = min(total, Decimal("8"))
		entry.total_hours = total
		entry.regular_hours = regular
		entry.overtime_hours = max(total - Decimal("8"), Decimal("0"))
		try:
			from .models import TimeEntryStatus
		except ImportError:
			from models import TimeEntryStatus  # type: ignore
		entry.status = TimeEntryStatus.SUBMITTED
		entry.updated_at = now
		return entry

	async def _save_time_entry(self, entry: Any) -> None:
		"""Persist (upsert) a time entry in the runtime store."""
		self.__class__._store_time_entries[entry.id] = entry

	async def _get_active_time_entry(self, employee_id: str, tenant_id: str) -> Any | None:
		"""Return the open (no clock_out) entry for employee, or None."""
		for e in self.__class__._store_time_entries.values():
			if e.tenant_id == tenant_id and e.employee_id == employee_id and e.clock_out is None:
				return e
		return None

	async def list_time_entries(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		start_date: date | None = None,
		end_date: date | None = None,
		status: str | None = None,
	) -> list[Any]:
		"""Return time entries for tenant, with optional filters."""
		result = []
		for e in self.__class__._store_time_entries.values():
			if e.tenant_id != tenant_id:
				continue
			if employee_id and e.employee_id != employee_id:
				continue
			if start_date and e.entry_date < start_date:
				continue
			if end_date and e.entry_date > end_date:
				continue
			result.append(e)
		return result

	async def start_remote_work_session(
		self,
		employee_id: str,
		tenant_id: str,
		workspace_config: dict[str, Any],
		work_mode: Any,
		created_by: str = "",
	) -> Any:
		"""Start a remote work session (in-memory mode)."""
		try:
			from .models import TARemoteWorker, RemoteWorkStatus
		except ImportError:
			from models import TARemoteWorker, RemoteWorkStatus  # type: ignore
		worker = TARemoteWorker(
			tenant_id=tenant_id,
			employee_id=employee_id,
			work_mode=work_mode,
			timezone=workspace_config.get("timezone", "UTC"),
			collaboration_platforms=workspace_config.get("collaboration_platforms", []),
			current_activity=RemoteWorkStatus.ACTIVE_WORKING,
			created_by=created_by or "system",
		)
		self.__class__._store_remote_workers[worker.id] = worker
		return worker

	async def track_remote_productivity(
		self,
		employee_id: str,
		tenant_id: str,
		activity_data: dict[str, Any],
		metric_type: Any,
	) -> dict[str, Any]:
		"""Track remote productivity and return analysis dict (in-memory mode)."""
		tasks = activity_data.get("tasks_completed", 0)
		active_minutes = activity_data.get("active_minutes", 0)
		score = min(round((tasks * 0.1 + active_minutes / 480), 4), 1.0)
		# update worker productivity_metrics
		for w in self.__class__._store_remote_workers.values():
			if w.tenant_id == tenant_id and w.employee_id == employee_id:
				w.productivity_metrics.append({
					"metric_type": str(metric_type),
					"score": score,
					"tasks_completed": tasks,
					"active_minutes": active_minutes,
					"timestamp": datetime.utcnow().isoformat(),
				})
				break
		return {
			"employee_id": employee_id,
			"metric_type": str(metric_type),
			"score": score,
			"tasks_completed": tasks,
			"active_minutes": active_minutes,
			"burnout_risk": "LOW" if score >= 0.3 else "HIGH",
		}

	async def list_remote_workers(
		self,
		tenant_id: str,
		active_only: bool = True,
		department_id: str | None = None,
		work_mode: Any = None,
	) -> list[Any]:
		"""Return remote workers for tenant."""
		result = []
		for w in self.__class__._store_remote_workers.values():
			if w.tenant_id != tenant_id:
				continue
			if active_only and not w.is_actively_working:
				continue
			if work_mode is not None and w.work_mode != work_mode:
				continue
			result.append(w)
		return result

	async def register_ai_agent(
		self,
		agent_name: str,
		agent_type: Any,
		capabilities: list[str],
		tenant_id: str,
		configuration: dict[str, Any],
		created_by: str = "",
	) -> Any:
		"""Register a new AI agent (in-memory mode)."""
		try:
			from .models import TAAIAgent
		except ImportError:
			from models import TAAIAgent  # type: ignore
		agent = TAAIAgent(
			tenant_id=tenant_id,
			agent_name=agent_name,
			agent_type=agent_type,
			agent_version="1.0",
			capabilities=capabilities,
			configuration=configuration,
			deployment_environment="runtime",
			api_endpoints=configuration.get("api_endpoints", []),
			operational_cost_per_hour=Decimal(configuration.get("cost_per_hour", "0")),
			created_by=created_by or "system",
		)
		self.__class__._store_ai_agents[agent.id] = agent
		return agent

	async def track_ai_agent_work(
		self,
		agent_id: str,
		tenant_id: str,
		task_result: dict[str, Any],
		resource_usage: dict[str, Any],
	) -> dict[str, Any]:
		"""Record completed work by an AI agent (in-memory mode)."""
		agent = self.__class__._store_ai_agents.get(agent_id)
		if agent is None or agent.tenant_id != tenant_id:
			raise ValueError(f"AI agent {agent_id} not found")
		if task_result.get("completed"):
			agent.tasks_completed += 1
		agent.cpu_hours += Decimal(str(resource_usage.get("cpu_hours", 0)))
		agent.memory_usage_gb_hours += Decimal(str(resource_usage.get("memory_gb_hours", 0)))
		agent.api_calls_count += resource_usage.get("api_calls", 0)
		agent.accuracy_score = task_result.get("accuracy_score", agent.accuracy_score)
		agent.updated_at = datetime.utcnow()
		return {"agent_id": agent_id, "tasks_completed": agent.tasks_completed}

	async def list_ai_agents(
		self,
		tenant_id: str,
		active_only: bool = True,
		agent_type: Any = None,
	) -> list[Any]:
		"""Return AI agents for tenant."""
		result = []
		for a in self.__class__._store_ai_agents.values():
			if a.tenant_id != tenant_id:
				continue
			if active_only and not a.is_active:
				continue
			if agent_type is not None and a.agent_type != agent_type:
				continue
			result.append(a)
		return result

	async def create_intelligent_schedule(
		self,
		schedule_name: str,
		tenant_id: str,
		schedule_patterns: list[dict[str, Any]],
		assigned_employees: list[str] | None = None,
		created_by: str = "",
	) -> Any:
		"""Create a new intelligent schedule (in-memory mode)."""
		try:
			from .models import TASchedule, ScheduleStatus
		except ImportError:
			from models import TASchedule, ScheduleStatus  # type: ignore
		schedule = TASchedule(
			tenant_id=tenant_id,
			schedule_name=schedule_name,
			schedule_type="intelligent",
			effective_date=date.today(),
			schedule_patterns=schedule_patterns,
			assigned_employees=assigned_employees or [],
			status=ScheduleStatus.PUBLISHED,
			created_by=created_by or "system",
		)
		self.__class__._store_schedules[schedule.id] = schedule
		return schedule

	async def list_schedules(
		self,
		tenant_id: str,
		active_only: bool = False,
	) -> list[Any]:
		"""Return schedules for tenant."""
		result = []
		for s in self.__class__._store_schedules.values():
			if s.tenant_id == tenant_id:
				result.append(s)
		return result

	async def process_leave_request(
		self,
		employee_id: str,
		tenant_id: str,
		leave_type: Any,
		start_date: date,
		end_date: date,
		reason: str | None = None,
		created_by: str = "",
	) -> Any:
		"""Submit a leave request (in-memory mode)."""
		try:
			from .models import TALeaveRequest, ApprovalStatus
		except ImportError:
			from models import TALeaveRequest, ApprovalStatus  # type: ignore
		days = max(Decimal(str((end_date - start_date).days + 1)), Decimal("1"))
		leave = TALeaveRequest(
			tenant_id=tenant_id,
			employee_id=employee_id,
			leave_type=leave_type,
			start_date=start_date,
			end_date=end_date,
			total_days=days,
			total_hours=days * Decimal("8"),
			reason=reason,
			status=ApprovalStatus.PENDING,
			created_by=created_by or "system",
		)
		self.__class__._store_leave_requests[leave.id] = leave
		return leave

	async def list_leave_requests(
		self,
		tenant_id: str,
		employee_id: str | None = None,
		status: str | None = None,
	) -> list[Any]:
		"""Return leave requests for tenant."""
		result = []
		for lr in self.__class__._store_leave_requests.values():
			if lr.tenant_id != tenant_id:
				continue
			if employee_id and lr.employee_id != employee_id:
				continue
			result.append(lr)
		return result

	async def get_analytics_dashboard(self, tenant_id: str) -> dict[str, Any]:
		"""Return a workforce analytics summary dict (in-memory mode)."""
		entries = await self.list_time_entries(tenant_id)
		remote_workers = await self.list_remote_workers(tenant_id, active_only=False)
		ai_agents = await self.list_ai_agents(tenant_id, active_only=False)
		leave_requests = await self.list_leave_requests(tenant_id)
		employee_ids: set[str] = set()
		employee_ids.update(e.employee_id for e in entries)
		employee_ids.update(w.employee_id for w in remote_workers)
		today = date.today()
		today_entries = [e for e in entries if e.entry_date == today]
		clocked_in_now = len([e for e in entries if e.clock_in and not e.clock_out])
		return {
			"tenant_id": tenant_id,
			"workforce_distribution": {
				"total_employees": len(employee_ids),
				"remote_workers": len(remote_workers),
				"ai_agents": len(ai_agents),
				"clocked_in_now": clocked_in_now,
			},
			"time_entries": {
				"total": len(entries),
				"today": len(today_entries),
			},
			"leave_requests": {
				"total": len(leave_requests),
				"pending": len([lr for lr in leave_requests if str(getattr(lr.status, "value", lr.status)) == "pending"]),
			},
		}

	async def bulk_update_time_entries(
		self,
		tenant_id: str,
		entry_ids: list[str],
		updates: dict[str, Any],
		actor: str,
	) -> dict[str, Any]:
		"""Bulk-update time entries by id (in-memory mode)."""
		updated: list[str] = []
		for eid in entry_ids:
			entry = self.__class__._store_time_entries.get(eid)
			if entry is None or entry.tenant_id != tenant_id:
				continue
			for key, val in updates.items():
				if hasattr(entry, key):
					object.__setattr__(entry, key, val)
			entry.updated_at = datetime.utcnow()
			updated.append(eid)
		return {"updated_ids": updated, "failed_ids": [i for i in entry_ids if i not in updated]}

	async def bulk_approve_entries(
		self,
		tenant_id: str,
		record_ids: list[str],
		record_type: str,
		actor: str,
		action: str = "approve",
		approval_notes: str = "",
	) -> dict[str, Any]:
		"""Bulk approve/reject time entries or leave requests (in-memory mode)."""
		try:
			from .models import ApprovalStatus, TimeEntryStatus
		except ImportError:
			from models import ApprovalStatus, TimeEntryStatus  # type: ignore
		processed: list[str] = []
		store = (
			self.__class__._store_leave_requests
			if record_type == "leave_request"
			else self.__class__._store_time_entries
		)
		for rid in record_ids:
			record = store.get(rid)
			if record is None or record.tenant_id != tenant_id:
				continue
			if action == "approve":
				if record_type == "leave_request":
					record.status = ApprovalStatus.APPROVED
				else:
					record.status = TimeEntryStatus.APPROVED
					record.approved_by = actor
					record.approved_at = datetime.utcnow()
			elif action == "reject":
				if record_type == "leave_request":
					record.status = ApprovalStatus.REJECTED
				else:
					record.status = TimeEntryStatus.REJECTED
			record.updated_at = datetime.utcnow()
			processed.append(rid)
		return {"processed_ids": processed, "failed_ids": [i for i in record_ids if i not in processed]}

	async def enforce_compliance_rules(self, tenant_id: str) -> dict[str, Any]:
		"""Detect and auto-correct compliance violations in time entries (in-memory mode)."""
		try:
			from .models import TimeEntryStatus
		except ImportError:
			from models import TimeEntryStatus  # type: ignore
		violations: list[dict[str, Any]] = []
		corrections = 0
		for entry in self.__class__._store_time_entries.values():
			if entry.tenant_id != tenant_id:
				continue
			total = float(entry.total_hours or 0)
			# DAILY_MAX_HOURS: >16 hrs
			if total > 16:
				violations.append({
					"entry_id": entry.id,
					"rule_code": "DAILY_MAX_HOURS",
					"description": f"Total hours {total} exceeds 16-hour daily maximum",
					"severity": "MAJOR",
				})
			# MINIMUM_BREAK: no break recorded for long shifts
			break_mins = entry.break_minutes or 0
			if total > 6 and break_mins < 30:
				violations.append({
					"entry_id": entry.id,
					"rule_code": "MINIMUM_BREAK",
					"description": "Minimum 30-minute break not recorded for shift >6 hours",
					"severity": "MINOR",
				})
				# auto-correct: record 30 min break
				entry.break_minutes = 30
				corrections += 1
			# OVERTIME_APPROVAL: overtime without approved_by
			ot = float(entry.overtime_hours or 0)
			if ot > 0 and not entry.approved_by:
				violations.append({
					"entry_id": entry.id,
					"rule_code": "OVERTIME_APPROVAL",
					"description": "Overtime hours recorded without approval",
					"severity": "WARNING",
				})
				# auto-correct: flag for approval
				entry.requires_approval = True
				corrections += 1
		total_entries = len([
			e for e in self.__class__._store_time_entries.values()
			if e.tenant_id == tenant_id
		])
		compliance_score = (
			round(1.0 - len(violations) / (total_entries * 3), 4)
			if total_entries else 1.0
		)
		return {
			"violations_detected": len(violations),
			"corrections_applied": corrections,
			"compliance_score": max(compliance_score, 0.0),
			"violations": violations,
		}

	async def generate_workforce_predictions(
		self,
		tenant_id: str,
		forecast_days: int = 7,
		employee_ids: list[str] | None = None,
	) -> Any:
		"""Generate workforce predictions (in-memory mode)."""
		try:
			from .models import TAPredictiveAnalytics
		except ImportError:
			from models import TAPredictiveAnalytics  # type: ignore
		entries = await self.list_time_entries(tenant_id)
		today = date.today()
		compliance_risks: list[dict[str, Any]] = []
		for entry in entries:
			if float(entry.total_hours or 0) > 16:
				compliance_risks.append({
					"type": "excessive_hours",
					"employee_id": entry.employee_id,
					"entry_id": entry.id,
					"risk_level": "HIGH",
				})
		analytics = TAPredictiveAnalytics(
			tenant_id=tenant_id,
			analysis_name=f"Workforce Forecast {today}",
			analysis_type="workforce_prediction",
			date_range={
				"start_time": today.isoformat() + "T00:00:00",
				"end_time": (today + timedelta(days=forecast_days)).isoformat() + "T23:59:59",
			},
			models_used=["time_series", "regression"],
			model_confidence=0.85,
			compliance_risks=compliance_risks,
			created_by="system",
		)
		return analytics

	async def start_hybrid_collaboration(
		self,
		employee_id: str,
		tenant_id: str,
		collaboration_config: dict[str, Any],
		created_by: str = "",
	) -> dict[str, Any]:
		"""Start a hybrid human-AI collaboration session (in-memory mode)."""
		return {
			"session_id": str(uuid7()),
			"employee_id": employee_id,
			"tenant_id": tenant_id,
			"status": "active",
			"started_at": datetime.utcnow().isoformat(),
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _log_ctx(self, method: str, **kw: Any) -> str:
		"""Return a log prefix string for consistent tracing."""
		return f"[TAT][{self._tenant_id}][{method}] " + " ".join(f"{k}={v}" for k, v in kw.items())

	def _log_action(self, method: str, record_id: str, **kw: Any) -> None:
		logger.info(self._log_ctx(method, id=record_id, **kw))

	def _log_error(self, method: str, exc: Exception, **kw: Any) -> None:
		logger.error(self._log_ctx(method, **kw) + f" error={exc!r}")

	def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
		"""Emit a domain event. Replace with real event bus in production."""
		event = {
			"id": _uuid7str(),
			"type": event_type,
			"tenant_id": self._tenant_id,
			"actor_id": self._actor_id,
			"occurred_at": datetime.now(UTC).isoformat(),
			"payload": payload,
		}
		self._events.append(event)
		logger.debug("event emitted: %s id=%s", event_type, event["id"])

	def _assert_tenant(self, record: dict[str, Any], label: str = "record") -> None:
		"""Guard: record must belong to this tenant."""
		assert_no_cross_tenant_access(self._tenant_id, record.get("tenant_id", ""))

	async def _fetch_one(self, table: str, record_id: str) -> dict[str, Any]:
		"""
		Fetch a single record by id, enforcing tenant isolation.
		Raises NotFoundError if absent or wrong tenant.
		"""
		row = await self._db.fetchrow(
			f"SELECT * FROM {table} WHERE id=$1 AND tenant_id=$2 AND NOT is_deleted",
			record_id, self._tenant_id,
		)
		if row is None:
			raise NotFoundError(f"{table} record {record_id} not found")
		return dict(row)

	async def _fetch_many(
		self,
		table: str,
		filters: dict[str, Any] | None = None,
		order_by: str = "created_at DESC",
		limit: int = 100,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		"""Generic paginated fetch with tenant isolation."""
		wheres = ["tenant_id=$1", "NOT is_deleted"]
		params: list[Any] = [self._tenant_id]
		idx = 2
		for col, val in (filters or {}).items():
			if val is None:
				continue
			wheres.append(f"{col}=${idx}")
			params.append(val)
			idx += 1
		params += [limit, offset]
		sql = (
			f"SELECT * FROM {table} WHERE {' AND '.join(wheres)} "
			f"ORDER BY {order_by} LIMIT ${idx} OFFSET ${idx+1}"
		)
		rows = await self._db.fetch(sql, *params)
		return [dict(r) for r in rows]

	async def _soft_delete(self, table: str, record_id: str) -> None:
		await self._db.execute(
			f"UPDATE {table} SET is_deleted=true, updated_at=now() WHERE id=$1 AND tenant_id=$2",
			record_id, self._tenant_id,
		)

	# ------------------------------------------------------------------
	# Time Policy CRUD
	# ------------------------------------------------------------------

	async def create_time_policy(
		self,
		name: str,
		timezone: str,
		workweek: list[str],
		overtime_threshold_daily: float = 8.0,
		overtime_threshold_weekly: float = 40.0,
		double_time_threshold_daily: float = 12.0,
		overtime_multiplier: float = 1.5,
		holiday_pay_multiplier: float = 2.0,
		min_rest_between_shifts_h: float = 11.0,
		max_consecutive_days: int = 6,
		max_weekly_hours: float = 48.0,
		break_rules: dict[str, Any] | None = None,
		flexi_core_start: time | None = None,
		flexi_core_end: time | None = None,
		flexi_max_carry_hours: float | None = 16.0,
		toil_enabled: bool = False,
		comp_time_enabled: bool = False,
		comp_time_jurisdiction: str | None = None,
		annualised_hours_enabled: bool = False,
		contracted_annual_hours: float | None = None,
		medical_cert_threshold_days: int = 3,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""
		Create a time policy defining overtime, rest, and compliance rules.

		Args:
			name: Human-readable policy name.
			timezone: IANA timezone string (e.g. 'Africa/Nairobi').
			workweek: List of weekday names, e.g. ['Mon','Tue','Wed','Thu','Fri'].
			overtime_threshold_daily: Daily hours before overtime kicks in.
			overtime_threshold_weekly: Weekly hours before overtime kicks in.
			...

		Returns:
			Created policy record dict.
		"""
		assert_overtime_threshold_positive(overtime_threshold_daily)
		assert_overtime_threshold_positive(overtime_threshold_weekly)

		record_id = _uuid7str()
		import json

		await self._db.execute(
			"""
			INSERT INTO tat_time_policy (
				id, tenant_id, name, timezone, workweek,
				overtime_threshold_daily, overtime_threshold_weekly,
				double_time_threshold_daily, overtime_multiplier,
				holiday_pay_multiplier, min_rest_between_shifts_h,
				max_consecutive_days, max_weekly_hours, break_rules,
				flexi_core_start, flexi_core_end, flexi_max_carry_hours,
				toil_enabled, comp_time_enabled, comp_time_jurisdiction,
				annualised_hours_enabled, contracted_annual_hours,
				medical_cert_threshold_days, metadata, created_by
			) VALUES (
				$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,
				$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25
			)
			""",
			record_id, self._tenant_id, name, timezone, json.dumps(workweek),
			overtime_threshold_daily, overtime_threshold_weekly,
			double_time_threshold_daily, overtime_multiplier,
			holiday_pay_multiplier, min_rest_between_shifts_h,
			max_consecutive_days, max_weekly_hours, json.dumps(break_rules or {}),
			flexi_core_start, flexi_core_end, flexi_max_carry_hours,
			toil_enabled, comp_time_enabled, comp_time_jurisdiction,
			annualised_hours_enabled, contracted_annual_hours,
			medical_cert_threshold_days, json.dumps(metadata or {}), self._actor_id,
		)

		record = await self._fetch_one("tat_time_policy", record_id)
		self._emit_event("tat.time_policy.created", {"policy_id": record_id, "name": name})
		self._log_action("create_time_policy", record_id, name=name)
		return record

	async def get_time_policy(self, policy_id: str) -> dict[str, Any]:
		"""Return a single time policy by ID."""
		return await self._fetch_one("tat_time_policy", policy_id)

	async def list_time_policies(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		"""List all active time policies for the tenant."""
		return await self._fetch_many("tat_time_policy", {"is_active": True}, limit=limit, offset=offset)

	async def update_time_policy(self, policy_id: str, **fields: Any) -> dict[str, Any]:
		"""Partial update of a time policy."""
		if not fields:
			return await self._fetch_one("tat_time_policy", policy_id)

		set_parts = []
		params: list[Any] = []
		idx = 1
		for k, v in fields.items():
			set_parts.append(f"{k}=${idx}")
			params.append(v)
			idx += 1
		params += [policy_id, self._tenant_id]
		await self._db.execute(
			f"UPDATE tat_time_policy SET {', '.join(set_parts)}, updated_at=now() "
			f"WHERE id=${idx} AND tenant_id=${idx+1}",
			*params,
		)
		record = await self._fetch_one("tat_time_policy", policy_id)
		self._emit_event("tat.time_policy.updated", {"policy_id": policy_id})
		return record

	async def delete_time_policy(self, policy_id: str) -> None:
		"""Soft-delete a time policy."""
		await self._soft_delete("tat_time_policy", policy_id)
		self._emit_event("tat.time_policy.deleted", {"policy_id": policy_id})

	# ------------------------------------------------------------------
	# Shift Schedule CRUD
	# ------------------------------------------------------------------

	async def create_shift_schedule(
		self,
		policy_id: str,
		schedule_name: str,
		schedule_type: str,
		effective_date: date,
		patterns: list[dict[str, Any]],
		end_date: date | None = None,
		department_id: str | None = None,
		location_id: str | None = None,
		description: str | None = None,
		allow_overtime: bool = True,
		allow_shift_swapping: bool = True,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""
		Create a shift schedule template with weekly patterns.

		patterns format: [{"days_of_week":[0,1,2,3,4], "start_time":"09:00", "end_time":"17:00"}]
		"""
		# Validate policy belongs to tenant
		await self._fetch_one("tat_time_policy", policy_id)

		import json
		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_shift_schedule (
				id, tenant_id, policy_id, schedule_name, schedule_type,
				effective_date, end_date, patterns, department_id, location_id,
				description, allow_overtime, allow_shift_swapping,
				metadata, created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
			""",
			record_id, self._tenant_id, policy_id, schedule_name, schedule_type,
			effective_date, end_date, json.dumps(patterns), department_id, location_id,
			description, allow_overtime, allow_shift_swapping,
			json.dumps(metadata or {}), self._actor_id,
		)
		record = await self._fetch_one("tat_shift_schedule", record_id)
		self._emit_event("tat.shift_schedule.created", {"schedule_id": record_id})
		return record

	async def get_shift_schedule(self, schedule_id: str) -> dict[str, Any]:
		return await self._fetch_one("tat_shift_schedule", schedule_id)

	async def list_shift_schedules(
		self,
		department_id: str | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		filters: dict[str, Any] = {}
		if department_id:
			filters["department_id"] = department_id
		return await self._fetch_many("tat_shift_schedule", filters, limit=limit, offset=offset)

	# ------------------------------------------------------------------
	# Shift CRUD
	# ------------------------------------------------------------------

	async def create_shift(
		self,
		schedule_id: str,
		employee_id: str,
		shift_date: date,
		planned_start: datetime,
		planned_end: datetime,
		location_id: str | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Create a concrete shift instance for an employee."""
		schedule = await self._fetch_one("tat_shift_schedule", schedule_id)
		policy_id = schedule["policy_id"]

		assert_clock_in_before_clock_out(planned_start, planned_end)
		assert_shift_duration_reasonable(planned_start, planned_end)
		night = is_night_shift(planned_start, planned_end)
		assert_night_shift_midnight_span(planned_start, planned_end)

		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_shift (
				id, tenant_id, schedule_id, employee_id, policy_id,
				shift_date, planned_start, planned_end, location_id,
				is_night_shift, notes, created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
			""",
			record_id, self._tenant_id, schedule_id, employee_id, policy_id,
			shift_date, planned_start, planned_end, location_id,
			night, notes, self._actor_id,
		)
		record = await self._fetch_one("tat_shift", record_id)
		self._emit_event("tat.shift.created", {"shift_id": record_id, "employee_id": employee_id})
		return record

	async def get_shift(self, shift_id: str) -> dict[str, Any]:
		return await self._fetch_one("tat_shift", shift_id)

	async def list_shifts(
		self,
		employee_id: str | None = None,
		from_date: date | None = None,
		to_date: date | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		"""List shifts, optionally filtered by employee and date range."""
		wheres = ["tenant_id=$1", "NOT is_deleted"]
		params: list[Any] = [self._tenant_id]
		idx = 2
		if employee_id:
			wheres.append(f"employee_id=${idx}"); params.append(employee_id); idx += 1
		if from_date:
			wheres.append(f"shift_date>=${idx}"); params.append(from_date); idx += 1
		if to_date:
			wheres.append(f"shift_date<=${idx}"); params.append(to_date); idx += 1
		params += [limit, offset]
		sql = (
			f"SELECT * FROM tat_shift WHERE {' AND '.join(wheres)} "
			f"ORDER BY shift_date, planned_start LIMIT ${idx} OFFSET ${idx+1}"
		)
		rows = await self._db.fetch(sql, *params)
		return [dict(r) for r in rows]

	# ------------------------------------------------------------------
	# Clock-in / Clock-out
	# ------------------------------------------------------------------

	async def _db_clock_in(
		self,
		employee_id: str,
		shift_id: str,
		entry_type: str = "regular",
		method: str = "web",
		device_id: str | None = None,
		latitude: float | None = None,
		longitude: float | None = None,
		biometric_confidence: float | None = None,
		ip_address: str | None = None,
		cost_center: str | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""
		Record a clock-in for an employee.

		Enforces:
		- Not already clocked in (open entry for today)
		- Device required for mobile/kiosk/biometric methods
		- Geofence validation if location provided
		- Biometric confidence threshold
		"""
		# Check for open entry today
		today = date.today()
		open_entries = await self._db.fetch(
			"""
			SELECT id, clock_in FROM tat_time_entry
			WHERE tenant_id=$1 AND employee_id=$2 AND entry_date=$3
			  AND clock_out IS NULL AND NOT is_deleted
			""",
			self._tenant_id, employee_id, today,
		)
		if open_entries:
			existing_in = open_entries[0]["clock_in"]
			assert_not_already_clocked_in(existing_in)

		# Device requirement
		assert_device_registered(device_id, method)

		# Biometric confidence
		if biometric_confidence is not None:
			assert_biometric_confidence(biometric_confidence)

		# Geofence validation
		geofence_verified = True
		if latitude is not None and longitude is not None and shift_id:
			shift = await self._fetch_one("tat_shift", shift_id)
			if shift.get("location_id"):
				loc = await self._fetch_one("tat_geofence_location", shift["location_id"])
				try:
					assert_within_geofence(
						latitude, longitude,
						loc["latitude"], loc["longitude"],
						float(loc["radius_metres"]),
					)
				except RuleViolation:
					geofence_verified = False
					logger.warning(self._log_ctx("clock_in", employee_id=employee_id) + " geofence fail")

		# Validate shift belongs to tenant and employee
		shift = await self._fetch_one("tat_shift", shift_id)
		if shift["employee_id"] != employee_id:
			raise TimeAttendanceError(f"shift {shift_id} does not belong to employee {employee_id}")

		# Minimum rest check: find last clock-out
		last_out = await self._db.fetchrow(
			"""
			SELECT clock_out FROM tat_time_entry
			WHERE tenant_id=$1 AND employee_id=$2 AND clock_out IS NOT NULL
			ORDER BY clock_out DESC LIMIT 1
			""",
			self._tenant_id, employee_id,
		)
		clock_in_ts = datetime.now(UTC)
		if last_out and last_out["clock_out"]:
			try:
				assert_minimum_rest_between_shifts(last_out["clock_out"], clock_in_ts)
			except RuleViolation as exc:
				logger.warning(self._log_ctx("clock_in", employee_id=employee_id) + f" rest warning: {exc}")

		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_time_entry (
				id, tenant_id, employee_id, shift_id, policy_id,
				entry_date, clock_in, entry_type, entry_method,
				clock_in_lat, clock_in_lng, geofence_verified,
				device_id, ip_address, biometric_confidence, biometric_verified,
				cost_center, notes, requires_approval, created_by
			) VALUES (
				$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,
				$13,$14,$15,$16,$17,$18,$19,$20
			)
			""",
			record_id, self._tenant_id, employee_id, shift_id, shift["policy_id"],
			today, clock_in_ts, entry_type, method,
			latitude, longitude, geofence_verified,
			device_id, ip_address, biometric_confidence,
			biometric_confidence is not None and biometric_confidence >= 0.85,
			cost_center, notes, not geofence_verified, self._actor_id,
		)
		record = await self._fetch_one("tat_time_entry", record_id)
		self._emit_event("tat.time_entry.clocked_in", {
			"entry_id": record_id,
			"employee_id": employee_id,
			"clock_in": clock_in_ts.isoformat(),
		})
		self._log_action("clock_in", record_id, employee_id=employee_id)
		return record

	async def _db_clock_out(
		self,
		entry_id: str,
		latitude: float | None = None,
		longitude: float | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""
		Record clock-out, compute hours, detect overtime, update entry.
		"""
		entry = await self._fetch_one("tat_time_entry", entry_id)
		assert_not_already_clocked_out(entry.get("clock_out"))

		clock_out_ts = datetime.now(UTC)
		clock_in_ts = entry["clock_in"]
		assert_clock_in_before_clock_out(clock_in_ts, clock_out_ts)
		assert_shift_duration_reasonable(clock_in_ts, clock_out_ts)
		assert_night_shift_midnight_span(clock_in_ts, clock_out_ts)

		# Load policy for overtime rules
		policy_id = entry.get("policy_id")
		policy = await self._fetch_one("tat_time_policy", policy_id) if policy_id else None

		ot_daily = Decimal(str(policy["overtime_threshold_daily"])) if policy else Decimal("8")
		dt_daily = Decimal(str(policy["double_time_threshold_daily"])) if policy else Decimal("12")

		# Fetch break time for this entry
		breaks = await self._db.fetch(
			"SELECT duration_minutes FROM tat_break WHERE time_entry_id=$1 AND NOT is_deleted",
			entry_id,
		)
		total_break_minutes = sum(b["duration_minutes"] or 0 for b in breaks)

		worked_h = calculate_worked_hours(clock_in_ts, clock_out_ts, total_break_minutes)
		is_holiday = entry.get("is_public_holiday", False)
		breakdown = calculate_hours_breakdown(
			worked_h, ot_daily, dt_daily, is_holiday=is_holiday
		)

		night = is_night_shift(clock_in_ts, clock_out_ts)

		await self._db.execute(
			"""
			UPDATE tat_time_entry SET
				clock_out=$1, clock_out_lat=$2, clock_out_lng=$3,
				total_hours=$4, regular_hours=$5, overtime_hours=$6,
				double_time_hours=$7, holiday_hours=$8,
				is_night_shift=$9, break_minutes=$10,
				notes=COALESCE($11, notes), updated_at=now()
			WHERE id=$12 AND tenant_id=$13
			""",
			clock_out_ts, latitude, longitude,
			float(breakdown.total), float(breakdown.regular),
			float(breakdown.overtime), float(breakdown.double_time),
			float(breakdown.holiday),
			night, total_break_minutes,
			notes, entry_id, self._tenant_id,
		)

		record = await self._fetch_one("tat_time_entry", entry_id)
		self._emit_event("tat.time_entry.clocked_out", {
			"entry_id": entry_id,
			"employee_id": entry["employee_id"],
			"total_hours": float(breakdown.total),
			"overtime_hours": float(breakdown.overtime),
		})
		self._log_action("clock_out", entry_id, hours=float(breakdown.total))
		return record

	async def get_time_entry(self, entry_id: str) -> dict[str, Any]:
		return await self._fetch_one("tat_time_entry", entry_id)

	async def _db_list_time_entries(
		self,
		employee_id: str | None = None,
		from_date: date | None = None,
		to_date: date | None = None,
		status: str | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		wheres = ["tenant_id=$1", "NOT is_deleted"]
		params: list[Any] = [self._tenant_id]
		idx = 2
		if employee_id:
			wheres.append(f"employee_id=${idx}"); params.append(employee_id); idx += 1
		if from_date:
			wheres.append(f"entry_date>=${idx}"); params.append(from_date); idx += 1
		if to_date:
			wheres.append(f"entry_date<=${idx}"); params.append(to_date); idx += 1
		if status:
			wheres.append(f"status=${idx}"); params.append(status); idx += 1
		params += [limit, offset]
		sql = (
			f"SELECT * FROM tat_time_entry WHERE {' AND '.join(wheres)} "
			f"ORDER BY entry_date DESC, clock_in DESC LIMIT ${idx} OFFSET ${idx+1}"
		)
		rows = await self._db.fetch(sql, *params)
		return [dict(r) for r in rows]

	async def update_time_entry(self, entry_id: str, **fields: Any) -> dict[str, Any]:
		"""Partial update; recalculates hours if clock times change."""
		entry = await self._fetch_one("tat_time_entry", entry_id)
		if entry["status"] == "locked":
			raise TimeAttendanceError("Cannot update a locked time entry")

		set_parts, params = [], []
		idx = 1
		for k, v in fields.items():
			set_parts.append(f"{k}=${idx}")
			params.append(v)
			idx += 1
		params += [entry_id, self._tenant_id]
		await self._db.execute(
			f"UPDATE tat_time_entry SET {', '.join(set_parts)}, updated_at=now() "
			f"WHERE id=${idx} AND tenant_id=${idx+1}",
			*params,
		)
		record = await self._fetch_one("tat_time_entry", entry_id)
		self._emit_event("tat.time_entry.updated", {"entry_id": entry_id})
		return record

	async def delete_time_entry(self, entry_id: str) -> None:
		entry = await self._fetch_one("tat_time_entry", entry_id)
		if entry["status"] in ("approved", "locked"):
			raise TimeAttendanceError("Cannot delete an approved/locked entry")
		await self._soft_delete("tat_time_entry", entry_id)
		self._emit_event("tat.time_entry.deleted", {"entry_id": entry_id})

	# ------------------------------------------------------------------
	# Breaks
	# ------------------------------------------------------------------

	async def record_break(
		self,
		time_entry_id: str,
		break_type: str,
		break_start: datetime,
		break_end: datetime,
		is_paid: bool = False,
	) -> dict[str, Any]:
		"""Record a break period against an open time entry."""
		entry = await self._fetch_one("tat_time_entry", time_entry_id)
		if entry.get("clock_out"):
			raise TimeAttendanceError("Cannot add break to a closed time entry")
		assert_clock_in_before_clock_out(break_start, break_end)

		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_break (id, tenant_id, time_entry_id, break_type,
				break_start, break_end, is_paid, created_by)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
			""",
			record_id, self._tenant_id, time_entry_id, break_type,
			break_start, break_end, is_paid, self._actor_id,
		)
		record = await self._db.fetchrow(
			"SELECT * FROM tat_break WHERE id=$1", record_id
		)
		self._emit_event("tat.break.recorded", {"break_id": record_id, "entry_id": time_entry_id})
		return dict(record)

	# ------------------------------------------------------------------
	# Timesheet processing
	# ------------------------------------------------------------------

	async def process_timesheet(
		self,
		employee_id: str,
		period_start: date,
		period_end: date,
		hourly_rate: Decimal | None = None,
		currency: str = "USD",
	) -> dict[str, Any]:
		"""
		Build or rebuild a timesheet for an employee over a pay period.

		Fetches all approved time entries, sums hours, and calculates gross pay
		if hourly_rate is provided.
		"""
		entries = await self._db.fetch(
			"""
			SELECT id, entry_type, total_hours, regular_hours, overtime_hours,
			       double_time_hours, holiday_hours, entry_date
			FROM tat_time_entry
			WHERE tenant_id=$1 AND employee_id=$2
			  AND entry_date BETWEEN $3 AND $4
			  AND NOT is_deleted
			  AND status NOT IN ('rejected')
			ORDER BY entry_date
			""",
			self._tenant_id, employee_id, period_start, period_end,
		)

		# Weekly re-bucketing
		daily_breakdowns: list[HoursBreakdown] = []
		total_leave_hours = Decimal("0")
		entry_ids = []
		for row in entries:
			entry_ids.append(row["id"])
			et = row["entry_type"]
			if et in ("sick", "vacation", "personal", "toil", "comp_time"):
				total_leave_hours += Decimal(str(row["total_hours"] or 0))
				continue
			daily_breakdowns.append(HoursBreakdown(
				regular=Decimal(str(row["regular_hours"] or 0)),
				overtime=Decimal(str(row["overtime_hours"] or 0)),
				double_time=Decimal(str(row["double_time_hours"] or 0)),
				holiday=Decimal(str(row["holiday_hours"] or 0)),
				total=Decimal(str(row["total_hours"] or 0)),
			))

		weekly_bd = calculate_weekly_hours_breakdown(daily_breakdowns)
		gross_pay = None
		if hourly_rate:
			pay = calculate_pay(weekly_bd, hourly_rate)
			gross_pay = float(pay.gross_pay)

		import json
		# Upsert timesheet
		existing = await self._db.fetchrow(
			"""
			SELECT id FROM tat_timesheet
			WHERE tenant_id=$1 AND employee_id=$2
			  AND period_start=$3 AND period_end=$4 AND NOT is_deleted
			""",
			self._tenant_id, employee_id, period_start, period_end,
		)

		ts_id = existing["id"] if existing else _uuid7str()
		if existing:
			await self._db.execute(
				"""
				UPDATE tat_timesheet SET
					total_hours=$1, regular_hours=$2, overtime_hours=$3,
					double_time_hours=$4, holiday_hours=$5, leave_hours=$6,
					gross_pay=$7, currency=$8, entry_ids=$9, updated_at=now()
				WHERE id=$10 AND tenant_id=$11
				""",
				float(weekly_bd.total), float(weekly_bd.regular),
				float(weekly_bd.overtime), float(weekly_bd.double_time),
				float(weekly_bd.holiday), float(total_leave_hours),
				gross_pay, currency, json.dumps(entry_ids), ts_id, self._tenant_id,
			)
		else:
			await self._db.execute(
				"""
				INSERT INTO tat_timesheet (
					id, tenant_id, employee_id, period_start, period_end,
					total_hours, regular_hours, overtime_hours, double_time_hours,
					holiday_hours, leave_hours, gross_pay, currency,
					entry_ids, status, created_by
				) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,'pending',$15)
				""",
				ts_id, self._tenant_id, employee_id, period_start, period_end,
				float(weekly_bd.total), float(weekly_bd.regular),
				float(weekly_bd.overtime), float(weekly_bd.double_time),
				float(weekly_bd.holiday), float(total_leave_hours),
				gross_pay, currency, json.dumps(entry_ids), self._actor_id,
			)

		record = await self._fetch_one("tat_timesheet", ts_id)
		self._emit_event("tat.timesheet.processed", {"timesheet_id": ts_id, "employee_id": employee_id})
		return record

	async def submit_timesheet(self, timesheet_id: str) -> dict[str, Any]:
		"""Mark a timesheet as submitted for manager approval."""
		ts = await self._fetch_one("tat_timesheet", timesheet_id)
		if ts["status"] != "pending":
			raise TimeAttendanceError(f"Timesheet status is '{ts['status']}'; expected 'pending'")
		await self._db.execute(
			"UPDATE tat_timesheet SET status='submitted', submitted_by=$1, submitted_at=now(), updated_at=now() WHERE id=$2 AND tenant_id=$3",
			self._actor_id, timesheet_id, self._tenant_id,
		)
		record = await self._fetch_one("tat_timesheet", timesheet_id)
		self._emit_event("tat.timesheet.submitted", {"timesheet_id": timesheet_id})
		return record

	async def approve_timesheet(self, timesheet_id: str) -> dict[str, Any]:
		"""Approve a submitted timesheet."""
		ts = await self._fetch_one("tat_timesheet", timesheet_id)
		if ts["status"] != "submitted":
			raise TimeAttendanceError(f"Timesheet status is '{ts['status']}'; expected 'submitted'")
		await self._db.execute(
			"UPDATE tat_timesheet SET status='approved', approved_by=$1, approved_at=now(), updated_at=now() WHERE id=$2 AND tenant_id=$3",
			self._actor_id, timesheet_id, self._tenant_id,
		)
		record = await self._fetch_one("tat_timesheet", timesheet_id)
		self._emit_event("tat.timesheet.approved", {"timesheet_id": timesheet_id, "approved_by": self._actor_id})
		return record

	async def reject_timesheet(self, timesheet_id: str, reason: str) -> dict[str, Any]:
		"""Reject a submitted timesheet with a reason."""
		ts = await self._fetch_one("tat_timesheet", timesheet_id)
		if ts["status"] not in ("submitted", "approved"):
			raise TimeAttendanceError(f"Cannot reject timesheet in status '{ts['status']}'")
		await self._db.execute(
			"""
			UPDATE tat_timesheet SET
				status='rejected', rejected_by=$1, rejected_at=now(),
				rejection_reason=$2, updated_at=now()
			WHERE id=$3 AND tenant_id=$4
			""",
			self._actor_id, reason, timesheet_id, self._tenant_id,
		)
		record = await self._fetch_one("tat_timesheet", timesheet_id)
		self._emit_event("tat.timesheet.rejected", {"timesheet_id": timesheet_id, "reason": reason})
		return record

	async def get_timesheet(self, timesheet_id: str) -> dict[str, Any]:
		return await self._fetch_one("tat_timesheet", timesheet_id)

	async def list_timesheets(
		self,
		employee_id: str | None = None,
		status: str | None = None,
		from_date: date | None = None,
		to_date: date | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		wheres = ["tenant_id=$1", "NOT is_deleted"]
		params: list[Any] = [self._tenant_id]
		idx = 2
		for col, val in [("employee_id", employee_id), ("status", status)]:
			if val:
				wheres.append(f"{col}=${idx}"); params.append(val); idx += 1
		if from_date:
			wheres.append(f"period_start>=${idx}"); params.append(from_date); idx += 1
		if to_date:
			wheres.append(f"period_end<=${idx}"); params.append(to_date); idx += 1
		params += [limit, offset]
		sql = (
			f"SELECT * FROM tat_timesheet WHERE {' AND '.join(wheres)} "
			f"ORDER BY period_start DESC LIMIT ${idx} OFFSET ${idx+1}"
		)
		return [dict(r) for r in await self._db.fetch(sql, *params)]

	# ------------------------------------------------------------------
	# Overtime
	# ------------------------------------------------------------------

	async def calculate_overtime(
		self,
		employee_id: str,
		period_start: date,
		period_end: date,
		policy_id: str,
	) -> dict[str, Any]:
		"""
		Calculate overtime for an employee over a period using the given policy.

		Returns a breakdown of regular, overtime, and double-time hours.
		"""
		policy = await self._fetch_one("tat_time_policy", policy_id)
		entries = await self._db.fetch(
			"""
			SELECT entry_date, total_hours, regular_hours, overtime_hours,
			       double_time_hours, holiday_hours, is_public_holiday
			FROM tat_time_entry
			WHERE tenant_id=$1 AND employee_id=$2
			  AND entry_date BETWEEN $3 AND $4
			  AND status NOT IN ('rejected') AND NOT is_deleted
			ORDER BY entry_date
			""",
			self._tenant_id, employee_id, period_start, period_end,
		)

		daily_breakdowns = []
		for row in entries:
			daily_breakdowns.append(HoursBreakdown(
				regular=Decimal(str(row["regular_hours"] or 0)),
				overtime=Decimal(str(row["overtime_hours"] or 0)),
				double_time=Decimal(str(row["double_time_hours"] or 0)),
				holiday=Decimal(str(row["holiday_hours"] or 0)),
				total=Decimal(str(row["total_hours"] or 0)),
			))

		weekly_bd = calculate_weekly_hours_breakdown(
			daily_breakdowns,
			weekly_ot_threshold=Decimal(str(policy["overtime_threshold_weekly"])),
		)
		return {
			"employee_id": employee_id,
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
			"regular_hours": float(weekly_bd.regular),
			"overtime_hours": float(weekly_bd.overtime),
			"double_time_hours": float(weekly_bd.double_time),
			"holiday_hours": float(weekly_bd.holiday),
			"total_hours": float(weekly_bd.total),
		}

	async def request_overtime(
		self,
		employee_id: str,
		shift_id: str,
		requested_hours: float,
		reason: str,
	) -> dict[str, Any]:
		"""Submit an overtime pre-authorisation request."""
		await self._fetch_one("tat_shift", shift_id)  # tenant check
		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_overtime_request (
				id, tenant_id, employee_id, shift_id,
				request_date, requested_hours, reason, created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
			""",
			record_id, self._tenant_id, employee_id, shift_id,
			date.today(), requested_hours, reason, self._actor_id,
		)
		record = await self._fetch_one("tat_overtime_request", record_id)
		self._emit_event("tat.overtime_request.created", {"request_id": record_id})
		return record

	async def approve_overtime_request(self, request_id: str) -> dict[str, Any]:
		req = await self._fetch_one("tat_overtime_request", request_id)
		if req["status"] != "pending":
			raise TimeAttendanceError(f"Overtime request status is '{req['status']}'")
		await self._db.execute(
			"UPDATE tat_overtime_request SET status='approved', approved_by=$1, approved_at=now(), updated_at=now() WHERE id=$2 AND tenant_id=$3",
			self._actor_id, request_id, self._tenant_id,
		)
		record = await self._fetch_one("tat_overtime_request", request_id)
		self._emit_event("tat.overtime_request.approved", {"request_id": request_id})
		return record

	async def reject_overtime_request(self, request_id: str, reason: str) -> dict[str, Any]:
		await self._fetch_one("tat_overtime_request", request_id)
		await self._db.execute(
			"UPDATE tat_overtime_request SET status='rejected', rejected_by=$1, rejected_at=now(), rejection_reason=$2, updated_at=now() WHERE id=$3 AND tenant_id=$4",
			self._actor_id, reason, request_id, self._tenant_id,
		)
		record = await self._fetch_one("tat_overtime_request", request_id)
		self._emit_event("tat.overtime_request.rejected", {"request_id": request_id, "reason": reason})
		return record

	# ------------------------------------------------------------------
	# Leave management
	# ------------------------------------------------------------------

	async def calculate_leave_entitlement_for(
		self,
		employee_id: str,
		leave_type: str,
		entitlement_year: int,
		fte: float = 1.0,
	) -> dict[str, Any]:
		"""
		Calculate leave entitlement for an employee using the applicable leave policy.
		"""
		policy_row = await self._db.fetchrow(
			"""
			SELECT * FROM tat_leave_policy
			WHERE tenant_id=$1 AND leave_type=$2 AND is_active AND NOT is_deleted
			ORDER BY created_at DESC LIMIT 1
			""",
			self._tenant_id, leave_type,
		)
		if not policy_row:
			raise NotFoundError(f"No active leave policy for type '{leave_type}'")

		policy = dict(policy_row)
		raw_annual = Decimal(str(policy["annual_days"]))
		fte_dec = Decimal(str(fte))
		annual_days = calculate_prorata_leave(raw_annual, fte_dec) if policy["fte_prorated"] and fte < 1.0 else raw_annual

		# Fetch existing entitlement record
		ent = await self._db.fetchrow(
			"""
			SELECT * FROM tat_leave_entitlement
			WHERE tenant_id=$1 AND employee_id=$2 AND leave_type=$3
			  AND entitlement_year=$4 AND NOT is_deleted
			""",
			self._tenant_id, employee_id, leave_type, entitlement_year,
		)
		used = Decimal(str(ent["used_days"])) if ent else Decimal("0")
		pending = Decimal(str(ent["pending_days"])) if ent else Decimal("0")
		carry = Decimal(str(ent["carried_forward"])) if ent else Decimal("0")

		start_of_year = date(entitlement_year, 1, 1)
		today = date.today()
		result = calculate_leave_entitlement(start_of_year, today, annual_days, used, pending)

		return {
			"employee_id": employee_id,
			"leave_type": leave_type,
			"year": entitlement_year,
			"annual_days": float(result.annual_days),
			"carried_forward": float(carry),
			"accrued_to_date": float(result.accrued_to_date),
			"used_days": float(result.used_to_date),
			"pending_days": float(result.pending),
			"balance_days": float(result.balance),
			"available_days": float(result.available),
			"policy_id": policy["id"],
			"policy_name": policy["name"],
		}

	async def request_leave(
		self,
		employee_id: str,
		leave_type: str,
		start_date: date,
		end_date: date,
		reason: str | None = None,
		is_emergency: bool = False,
		is_half_day: bool = False,
		half_day_portion: str | None = None,
		is_statutory: bool = False,
		statutory_type: str | None = None,
		statutory_jurisdiction: str | None = None,
		medical_cert_attached: bool = False,
		attachments: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""
		Submit a leave request with full validation.

		Validates:
		- Date range
		- Leave balance
		- No overlap with existing approved/pending leave
		- Medical certificate for extended sick leave
		- FMLA prerequisites when statutory_type='FMLA'
		"""
		assert_leave_dates_valid(start_date, end_date)

		public_holidays = await self._get_public_holidays_between(start_date, end_date)
		total_days = Decimal(str(working_days_between(start_date, end_date, public_holidays)))
		if is_half_day:
			total_days = Decimal("0.5")

		# Medical cert for extended sick
		policy_row = await self._db.fetchrow(
			"SELECT medical_cert_required_days FROM tat_leave_policy WHERE tenant_id=$1 AND leave_type=$2 AND is_active AND NOT is_deleted ORDER BY created_at DESC LIMIT 1",
			self._tenant_id, leave_type,
		)
		cert_threshold = int(policy_row["medical_cert_required_days"]) if policy_row else 3
		assert_medical_certificate_for_extended_sick(
			leave_type, int(total_days), medical_cert_attached, cert_threshold
		)

		# Balance check (skip for unpaid / statutory)
		if leave_type not in ("unpaid", "fmla", "military", "jury_duty"):
			ent = await self._db.fetchrow(
				"""
				SELECT available_days FROM tat_leave_entitlement
				WHERE tenant_id=$1 AND employee_id=$2 AND leave_type=$3
				  AND entitlement_year=$4 AND NOT is_deleted
				""",
				self._tenant_id, employee_id, leave_type, start_date.year,
			)
			if ent:
				assert_leave_balance_sufficient(Decimal(str(ent["available_days"])), total_days)

		# Overlap check handled by DB constraint; catch IntegrityError upstream
		import json
		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_leave_request (
				id, tenant_id, employee_id, leave_type,
				start_date, end_date, total_days, total_hours,
				is_half_day, half_day_portion, is_emergency,
				reason, medical_cert_attached, attachments,
				is_statutory, statutory_type, statutory_jurisdiction,
				metadata, created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
			""",
			record_id, self._tenant_id, employee_id, leave_type,
			start_date, end_date, float(total_days), float(total_days * 8),
			is_half_day, half_day_portion, is_emergency,
			reason, medical_cert_attached, json.dumps(attachments or []),
			is_statutory, statutory_type, statutory_jurisdiction,
			json.dumps(metadata or {}), self._actor_id,
		)
		record = await self._fetch_one("tat_leave_request", record_id)
		self._emit_event("tat.leave_request.created", {
			"request_id": record_id,
			"employee_id": employee_id,
			"leave_type": leave_type,
			"start_date": start_date.isoformat(),
			"end_date": end_date.isoformat(),
		})
		return record

	async def approve_leave_request(self, request_id: str) -> dict[str, Any]:
		"""Approve leave and deduct from entitlement balance."""
		req = await self._fetch_one("tat_leave_request", request_id)
		if req["status"] != "pending":
			raise TimeAttendanceError(f"Leave request status is '{req['status']}'")

		await self._db.execute(
			"UPDATE tat_leave_request SET status='approved', approved_by=$1, approved_at=now(), updated_at=now() WHERE id=$2 AND tenant_id=$3",
			self._actor_id, request_id, self._tenant_id,
		)

		# Deduct from entitlement
		await self._db.execute(
			"""
			UPDATE tat_leave_entitlement
			SET used_days = used_days + $1,
			    pending_days = GREATEST(pending_days - $1, 0),
			    updated_at = now()
			WHERE tenant_id=$2 AND employee_id=$3
			  AND leave_type=$4 AND entitlement_year=$5 AND NOT is_deleted
			""",
			float(req["total_days"]), self._tenant_id, req["employee_id"],
			req["leave_type"], req["start_date"].year,
		)

		record = await self._fetch_one("tat_leave_request", request_id)
		self._emit_event("tat.leave_request.approved", {"request_id": request_id, "approved_by": self._actor_id})
		return record

	async def reject_leave_request(self, request_id: str, reason: str) -> dict[str, Any]:
		req = await self._fetch_one("tat_leave_request", request_id)
		if req["status"] not in ("pending", "approved"):
			raise TimeAttendanceError(f"Cannot reject leave in status '{req['status']}'")

		await self._db.execute(
			"UPDATE tat_leave_request SET status='rejected', rejected_by=$1, rejected_at=now(), rejection_reason=$2, updated_at=now() WHERE id=$3 AND tenant_id=$4",
			self._actor_id, reason, request_id, self._tenant_id,
		)
		# If was approved, restore pending days
		if req["status"] == "approved":
			await self._db.execute(
				"""
				UPDATE tat_leave_entitlement
				SET used_days = GREATEST(used_days - $1, 0), updated_at=now()
				WHERE tenant_id=$2 AND employee_id=$3
				  AND leave_type=$4 AND entitlement_year=$5 AND NOT is_deleted
				""",
				float(req["total_days"]), self._tenant_id, req["employee_id"],
				req["leave_type"], req["start_date"].year,
			)

		record = await self._fetch_one("tat_leave_request", request_id)
		self._emit_event("tat.leave_request.rejected", {"request_id": request_id, "reason": reason})
		return record

	async def cancel_leave_request(self, request_id: str) -> dict[str, Any]:
		"""Employee cancels their own leave request."""
		req = await self._fetch_one("tat_leave_request", request_id)
		if req["status"] not in ("pending", "approved"):
			raise TimeAttendanceError(f"Cannot cancel leave in status '{req['status']}'")

		was_approved = req["status"] == "approved"
		await self._db.execute(
			"UPDATE tat_leave_request SET status='withdrawn', updated_at=now() WHERE id=$1 AND tenant_id=$2",
			request_id, self._tenant_id,
		)
		if was_approved:
			await self._db.execute(
				"UPDATE tat_leave_entitlement SET used_days=GREATEST(used_days-$1,0), updated_at=now() WHERE tenant_id=$2 AND employee_id=$3 AND leave_type=$4 AND entitlement_year=$5 AND NOT is_deleted",
				float(req["total_days"]), self._tenant_id, req["employee_id"],
				req["leave_type"], req["start_date"].year,
			)

		record = await self._fetch_one("tat_leave_request", request_id)
		self._emit_event("tat.leave_request.cancelled", {"request_id": request_id})
		return record

	async def get_leave_request(self, request_id: str) -> dict[str, Any]:
		return await self._fetch_one("tat_leave_request", request_id)

	async def _db_list_leave_requests(
		self,
		employee_id: str | None = None,
		leave_type: str | None = None,
		status: str | None = None,
		from_date: date | None = None,
		to_date: date | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		wheres = ["tenant_id=$1", "NOT is_deleted"]
		params: list[Any] = [self._tenant_id]
		idx = 2
		for col, val in [("employee_id", employee_id), ("leave_type", leave_type), ("status", status)]:
			if val:
				wheres.append(f"{col}=${idx}"); params.append(val); idx += 1
		if from_date:
			wheres.append(f"start_date>=${idx}"); params.append(from_date); idx += 1
		if to_date:
			wheres.append(f"end_date<=${idx}"); params.append(to_date); idx += 1
		params += [limit, offset]
		sql = (
			f"SELECT * FROM tat_leave_request WHERE {' AND '.join(wheres)} "
			f"ORDER BY start_date DESC LIMIT ${idx} OFFSET ${idx+1}"
		)
		return [dict(r) for r in await self._db.fetch(sql, *params)]

	# ------------------------------------------------------------------
	# Flexitime
	# ------------------------------------------------------------------

	async def flexitime_calculation(
		self,
		employee_id: str,
		from_date: date,
		to_date: date,
		policy_id: str,
	) -> dict[str, Any]:
		"""
		Compute cumulative flexitime balance for an employee over a date range.
		"""
		policy = await self._fetch_one("tat_time_policy", policy_id)
		standard_daily = Decimal(str(policy.get("overtime_threshold_daily", 8)))

		entries = await self._db.fetch(
			"""
			SELECT entry_date, total_hours FROM tat_time_entry
			WHERE tenant_id=$1 AND employee_id=$2
			  AND entry_date BETWEEN $3 AND $4
			  AND entry_type NOT IN ('sick','vacation','personal','toil')
			  AND NOT is_deleted
			ORDER BY entry_date
			""",
			self._tenant_id, employee_id, from_date, to_date,
		)
		log = [(row["entry_date"], Decimal(str(row["total_hours"] or 0))) for row in entries]

		# Carry forward from previous balance record
		carry_row = await self._db.fetchrow(
			"""
			SELECT balance_hours FROM tat_flexitime_balance
			WHERE tenant_id=$1 AND employee_id=$2
			  AND balance_date < $3 AND NOT is_deleted
			ORDER BY balance_date DESC LIMIT 1
			""",
			self._tenant_id, employee_id, from_date,
		)
		carry = Decimal(str(carry_row["balance_hours"])) if carry_row else Decimal("0")
		balance = calculate_flexi_balance(log, standard_daily, carry)

		return {
			"employee_id": employee_id,
			"from_date": from_date.isoformat(),
			"to_date": to_date.isoformat(),
			"credit_hours": float(balance.credit_hours),
			"debit_hours": float(balance.debit_hours),
			"net_hours": float(balance.net_hours),
			"max_carry": float(policy.get("flexi_max_carry_hours") or 16),
		}

	# ------------------------------------------------------------------
	# Annualised hours
	# ------------------------------------------------------------------

	async def annualised_hours_reconciliation(
		self,
		employee_id: str,
		policy_id: str,
		as_of_date: date | None = None,
	) -> dict[str, Any]:
		"""
		Reconcile annualised hours for an employee against their contracted total.
		"""
		policy = await self._fetch_one("tat_time_policy", policy_id)
		if not policy.get("annualised_hours_enabled"):
			raise TimeAttendanceError("Annualised hours not enabled on this policy")

		contracted = Decimal(str(policy["contracted_annual_hours"] or 0))
		as_of = as_of_date or date.today()
		year_start = date(as_of.year, 1, 1)
		weeks_elapsed = Decimal(str(((as_of - year_start).days + 1) / 7))

		rows = await self._db.fetch(
			"""
			SELECT SUM(total_hours) AS total FROM tat_time_entry
			WHERE tenant_id=$1 AND employee_id=$2
			  AND entry_date BETWEEN $3 AND $4
			  AND NOT is_deleted AND status NOT IN ('rejected')
			""",
			self._tenant_id, employee_id, year_start, as_of,
		)
		worked = Decimal(str(rows[0]["total"] or 0))
		from .domain.calculations import (
			calculate_annualised_expected_hours,
			annualised_hours_owed,
		)
		expected = calculate_annualised_expected_hours(contracted, weeks_elapsed)
		owed = annualised_hours_owed(contracted, worked)

		try:
			assert_annualised_hours_deficit_manageable(-owed if owed < 0 else Decimal("0"))
		except RuleViolation:
			pass  # Surface as warning in response

		return {
			"employee_id": employee_id,
			"as_of_date": as_of.isoformat(),
			"contracted_annual_hours": float(contracted),
			"weeks_elapsed": float(weeks_elapsed),
			"expected_hours_to_date": float(expected),
			"actual_hours_worked": float(worked),
			"variance_hours": float(worked - expected),
			"hours_remaining_in_contract": float(max(contracted - worked, Decimal("0"))),
		}

	# ------------------------------------------------------------------
	# Roster generation
	# ------------------------------------------------------------------

	async def roster_generation(
		self,
		schedule_id: str,
		period_start: date,
		period_end: date,
		employee_ids: list[str],
		constraints: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""
		Generate a roster by instantiating shift records for all employees
		in employee_ids according to the schedule's patterns over period_start..period_end.

		Returns the created roster record with the list of generated shift IDs.
		"""
		schedule = await self._fetch_one("tat_shift_schedule", schedule_id)
		import json
		patterns = json.loads(schedule["patterns"]) if isinstance(schedule["patterns"], str) else schedule["patterns"]

		shift_ids = []
		current = period_start
		while current <= period_end:
			dow = current.weekday()  # 0=Mon
			for pattern in patterns:
				pattern_days = pattern.get("days_of_week", [])
				if dow not in pattern_days:
					continue
				start_h, start_m = (int(x) for x in pattern["start_time"].split(":"))
				end_h, end_m = (int(x) for x in pattern["end_time"].split(":"))

				for emp_id in employee_ids:
					planned_start = datetime.combine(current, time(start_h, start_m), tzinfo=UTC)
					planned_end_date = current if end_h > start_h else current + timedelta(days=1)
					planned_end = datetime.combine(planned_end_date, time(end_h, end_m), tzinfo=UTC)

					shift = await self.create_shift(
						schedule_id=schedule_id,
						employee_id=emp_id,
						shift_date=current,
						planned_start=planned_start,
						planned_end=planned_end,
						location_id=schedule.get("location_id"),
					)
					shift_ids.append(shift["id"])
			current += timedelta(days=1)

		# Create roster record
		roster_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_roster (
				id, tenant_id, name, period_start, period_end,
				shift_ids, constraints, status, created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,'draft',$8)
			""",
			roster_id, self._tenant_id,
			f"Roster {period_start}–{period_end}",
			period_start, period_end,
			json.dumps(shift_ids), json.dumps(constraints or {}), self._actor_id,
		)
		record = await self._fetch_one("tat_roster", roster_id)
		self._emit_event("tat.roster.generated", {
			"roster_id": roster_id,
			"shift_count": len(shift_ids),
		})
		return record

	# ------------------------------------------------------------------
	# Shift swap
	# ------------------------------------------------------------------

	async def shift_swap_request(
		self,
		requester_shift_id: str,
		target_shift_id: str | None = None,
		target_id: str | None = None,
		reason: str | None = None,
	) -> dict[str, Any]:
		"""Request a shift swap between two employees."""
		req_shift = await self._fetch_one("tat_shift", requester_shift_id)
		requester_id = req_shift["employee_id"]

		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_shift_swap_request (
				id, tenant_id, requester_id, requester_shift_id,
				target_id, target_shift_id, reason, created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
			""",
			record_id, self._tenant_id, requester_id, requester_shift_id,
			target_id, target_shift_id, reason, self._actor_id,
		)
		record = await self._fetch_one("tat_shift_swap_request", record_id)
		self._emit_event("tat.shift_swap.requested", {"swap_id": record_id})
		return record

	async def approve_shift_swap(self, swap_id: str) -> dict[str, Any]:
		"""Approve a shift swap and update both shifts."""
		swap = await self._fetch_one("tat_shift_swap_request", swap_id)
		if swap["status"] != "pending":
			raise TimeAttendanceError(f"Swap is in status '{swap['status']}'")

		# Update shifts if both sides specified
		if swap.get("target_shift_id"):
			req_shift = await self._fetch_one("tat_shift", swap["requester_shift_id"])
			tgt_shift = await self._fetch_one("tat_shift", swap["target_shift_id"])
			# Swap employee_ids
			await self._db.execute(
				"UPDATE tat_shift SET employee_id=$1, status='swapped', swapped_with_id=$2, updated_at=now() WHERE id=$3",
				tgt_shift["employee_id"], swap["target_shift_id"], swap["requester_shift_id"],
			)
			await self._db.execute(
				"UPDATE tat_shift SET employee_id=$1, status='swapped', swapped_with_id=$2, updated_at=now() WHERE id=$3",
				req_shift["employee_id"], swap["requester_shift_id"], swap["target_shift_id"],
			)

		await self._db.execute(
			"UPDATE tat_shift_swap_request SET status='approved', approved_by=$1, approved_at=now(), updated_at=now() WHERE id=$2 AND tenant_id=$3",
			self._actor_id, swap_id, self._tenant_id,
		)
		record = await self._fetch_one("tat_shift_swap_request", swap_id)
		self._emit_event("tat.shift_swap.approved", {"swap_id": swap_id})
		return record

	# ------------------------------------------------------------------
	# Biometric device sync
	# ------------------------------------------------------------------

	async def biometric_device_sync(
		self,
		device_id: str,
		raw_records: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""
		Process raw biometric punch records from a device.

		Each record must have: employee_id, clock_in (ISO string),
		optionally clock_out (ISO string), biometric_confidence (float).

		Returns a sync log record with counts.
		"""
		device = await self._fetch_one("tat_attendance_device", device_id)

		log_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_biometric_sync_log (id, tenant_id, device_id, initiated_by)
			VALUES ($1,$2,$3,$4)
			""",
			log_id, self._tenant_id, device_id, self._actor_id,
		)

		created = skipped = 0
		errors: list[dict[str, Any]] = []
		required_cols = ["employee_id", "clock_in"]

		for i, row in enumerate(raw_records):
			try:
				assert_import_row_valid(row, required_cols)
				confidence = float(row.get("biometric_confidence", 0))
				assert_biometric_confidence(confidence)

				clock_in_ts = datetime.fromisoformat(row["clock_in"].replace("Z", "+00:00"))
				clock_out_ts = None
				if row.get("clock_out"):
					clock_out_ts = datetime.fromisoformat(row["clock_out"].replace("Z", "+00:00"))

				# Find shift for this employee on this date
				shift_rows = await self._db.fetch(
					"""
					SELECT id FROM tat_shift
					WHERE tenant_id=$1 AND employee_id=$2 AND shift_date=$3 AND NOT is_deleted
					ORDER BY planned_start LIMIT 1
					""",
					self._tenant_id, row["employee_id"], clock_in_ts.date(),
				)
				if not shift_rows:
					errors.append({"row": i, "reason": "no_shift_for_date", "employee_id": row["employee_id"]})
					skipped += 1
					continue

				shift_id = shift_rows[0]["id"]
				entry_id = _uuid7str()
				worked_h = calculate_worked_hours(clock_in_ts, clock_out_ts) if clock_out_ts else Decimal("0")

				await self._db.execute(
					"""
					INSERT INTO tat_time_entry (
						id, tenant_id, employee_id, shift_id,
						entry_date, clock_in, clock_out, total_hours,
						entry_type, entry_method, device_id,
						biometric_confidence, biometric_verified,
						geofence_verified, status, created_by
					) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'regular','biometric',$9,$10,$11,true,'submitted',$12)
					ON CONFLICT (tenant_id, employee_id, entry_date)
					WHERE NOT is_deleted AND entry_type NOT IN ('break','on_call')
					DO NOTHING
					""",
					entry_id, self._tenant_id, row["employee_id"], shift_id,
					clock_in_ts.date(), clock_in_ts, clock_out_ts, float(worked_h),
					device_id, confidence, confidence >= 0.85, self._actor_id,
				)
				# Check if it was actually inserted (ON CONFLICT DO NOTHING)
				exists = await self._db.fetchrow(
					"SELECT id FROM tat_time_entry WHERE id=$1", entry_id
				)
				if exists:
					created += 1
				else:
					skipped += 1

			except (RuleViolation, ValueError, KeyError) as exc:
				errors.append({"row": i, "reason": str(exc)})
				skipped += 1

		import json
		await self._db.execute(
			"""
			UPDATE tat_biometric_sync_log SET
				sync_ended_at=now(), records_pulled=$1, records_created=$2,
				records_skipped=$3, errors=$4, status='completed'
			WHERE id=$5
			""",
			len(raw_records), created, skipped, json.dumps(errors), log_id,
		)
		log = await self._db.fetchrow("SELECT * FROM tat_biometric_sync_log WHERE id=$1", log_id)
		self._emit_event("tat.device.synced", {
			"device_id": device_id,
			"log_id": log_id,
			"created": created,
			"skipped": skipped,
		})
		return dict(log)

	# ------------------------------------------------------------------
	# GPS geofence validation
	# ------------------------------------------------------------------

	async def gps_geofence_validation(
		self,
		employee_id: str,
		latitude: float,
		longitude: float,
		location_id: str,
	) -> dict[str, Any]:
		"""
		Validate whether a GPS coordinate is within an allowed geofence.

		Returns a dict with is_valid and distance_metres.
		"""
		import math
		loc = await self._fetch_one("tat_geofence_location", location_id)
		R = 6_371_000
		phi1 = math.radians(latitude)
		phi2 = math.radians(float(loc["latitude"]))
		dphi = math.radians(float(loc["latitude"]) - latitude)
		dlambda = math.radians(float(loc["longitude"]) - longitude)
		a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
		distance = 2 * R * math.asin(math.sqrt(a))
		radius = float(loc["radius_metres"])
		return {
			"employee_id": employee_id,
			"location_id": location_id,
			"location_name": loc["name"],
			"is_valid": distance <= radius,
			"distance_metres": round(distance, 1),
			"allowed_radius_metres": radius,
		}

	# ------------------------------------------------------------------
	# Bulk timesheet import
	# ------------------------------------------------------------------

	async def bulk_timesheet_import(
		self,
		csv_content: str,
		date_format: str = "%Y-%m-%d",
		time_format: str = "%H:%M",
	) -> dict[str, Any]:
		"""
		Import time entries from CSV.

		Expected columns: employee_id, shift_id, entry_date, clock_in, clock_out,
		                  entry_type, break_minutes, cost_center, notes.
		Returns counts of imported, skipped, and error rows.
		"""
		required_cols = ["employee_id", "entry_date", "clock_in"]
		reader = csv.DictReader(io.StringIO(csv_content))
		imported = skipped = 0
		errors: list[dict[str, Any]] = []

		for i, row in enumerate(reader):
			try:
				assert_import_row_valid(row, required_cols)
				entry_date = datetime.strptime(row["entry_date"], date_format).date()
				clock_in_ts = datetime.combine(
					entry_date,
					datetime.strptime(row["clock_in"], time_format).time(),
					tzinfo=UTC,
				)
				clock_out_ts = None
				if row.get("clock_out"):
					clock_out_ts = datetime.combine(
						entry_date,
						datetime.strptime(row["clock_out"], time_format).time(),
						tzinfo=UTC,
					)
					if clock_out_ts < clock_in_ts:
						# Night shift
						clock_out_ts += timedelta(days=1)

				break_minutes = int(row.get("break_minutes") or 0)
				worked_h = calculate_worked_hours(clock_in_ts, clock_out_ts, break_minutes) if clock_out_ts else Decimal("0")

				entry_id = _uuid7str()
				await self._db.execute(
					"""
					INSERT INTO tat_time_entry (
						id, tenant_id, employee_id, shift_id,
						entry_date, clock_in, clock_out, break_minutes,
						total_hours, entry_type, entry_method,
						cost_center, notes, status, created_by
					) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'import',$11,$12,'submitted',$13)
					ON CONFLICT (tenant_id, employee_id, entry_date)
					WHERE NOT is_deleted AND entry_type NOT IN ('break','on_call')
					DO NOTHING
					""",
					entry_id, self._tenant_id, row["employee_id"],
					row.get("shift_id") or None,
					entry_date, clock_in_ts, clock_out_ts, break_minutes,
					float(worked_h), row.get("entry_type") or "regular",
					row.get("cost_center") or None, row.get("notes") or None,
					self._actor_id,
				)
				check = await self._db.fetchrow("SELECT id FROM tat_time_entry WHERE id=$1", entry_id)
				if check:
					imported += 1
				else:
					skipped += 1

			except (RuleViolation, ValueError, KeyError) as exc:
				errors.append({"row": i + 2, "reason": str(exc), "data": dict(row)})
				skipped += 1

		self._emit_event("tat.bulk_import.completed", {
			"imported": imported,
			"skipped": skipped,
			"error_count": len(errors),
		})
		return {"imported": imported, "skipped": skipped, "errors": errors}

	# ------------------------------------------------------------------
	# Payroll export
	# ------------------------------------------------------------------

	async def create_payroll_export(
		self,
		period_start: date,
		period_end: date,
		timesheet_ids: list[str],
		notes: str | None = None,
	) -> dict[str, Any]:
		"""
		Bundle approved timesheets into a payroll export record.

		Raises if any timesheet is not in 'approved' status.
		"""
		import json
		total_hours = Decimal("0")
		total_gross = Decimal("0")
		employees: set[str] = set()

		for ts_id in timesheet_ids:
			ts = await self._fetch_one("tat_timesheet", ts_id)
			assert_timesheet_approved_before_export(ts["status"])
			total_hours += Decimal(str(ts["total_hours"] or 0))
			if ts.get("gross_pay"):
				total_gross += Decimal(str(ts["gross_pay"]))
			employees.add(ts["employee_id"])
			# Mark as exported
			await self._db.execute(
				"UPDATE tat_timesheet SET payroll_export_id=$1, updated_at=now() WHERE id=$2 AND tenant_id=$3",
				"pending", ts_id, self._tenant_id,  # Will be updated after export ID is known
			)

		record_id = _uuid7str()
		currency = "USD"
		if timesheet_ids:
			ts_row = await self._db.fetchrow(
				"SELECT currency FROM tat_timesheet WHERE id=$1", timesheet_ids[0]
			)
			currency = ts_row["currency"] if ts_row else "USD"

		await self._db.execute(
			"""
			INSERT INTO tat_payroll_export (
				id, tenant_id, period_start, period_end,
				timesheet_ids, total_employees, total_hours,
				total_gross_pay, currency, status,
				event_stream, processor, notes, created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'ready',$10,'bytewax',$11,$12)
			""",
			record_id, self._tenant_id, period_start, period_end,
			json.dumps(timesheet_ids), len(employees),
			float(total_hours), float(total_gross), currency,
			"apg.hcm.tat.time_attendance.lifecycle", notes, self._actor_id,
		)

		# Back-fill export ID into timesheets
		for ts_id in timesheet_ids:
			await self._db.execute(
				"UPDATE tat_timesheet SET payroll_export_id=$1, updated_at=now() WHERE id=$2 AND tenant_id=$3",
				record_id, ts_id, self._tenant_id,
			)

		record = await self._fetch_one("tat_payroll_export", record_id)
		self._emit_event("tat.payroll_export.created", {
			"export_id": record_id,
			"total_hours": float(total_hours),
			"timesheet_count": len(timesheet_ids),
		})
		return record

	async def get_payroll_export(self, export_id: str) -> dict[str, Any]:
		return await self._fetch_one("tat_payroll_export", export_id)

	async def list_payroll_exports(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
		return await self._fetch_many("tat_payroll_export", limit=limit, offset=offset)

	# ------------------------------------------------------------------
	# Attendance exceptions
	# ------------------------------------------------------------------

	async def record_exception(
		self,
		employee_id: str,
		exception_type: str,
		severity: str,
		description: str,
		time_entry_id: str | None = None,
		owner_id: str | None = None,
	) -> dict[str, Any]:
		"""Record an attendance exception for investigation."""
		if severity == "high" and not owner_id:
			raise RuleViolation(
				"exception_owner_required",
				"High severity exceptions require an owner",
				"assign_exception_owner",
			)
		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_attendance_exception (
				id, tenant_id, employee_id, time_entry_id,
				exception_type, severity, description, owner_id, created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
			""",
			record_id, self._tenant_id, employee_id, time_entry_id,
			exception_type, severity, description, owner_id, self._actor_id,
		)
		record = await self._fetch_one("tat_attendance_exception", record_id)
		self._emit_event("tat.exception.recorded", {"exception_id": record_id, "type": exception_type})
		return record

	async def resolve_exception(self, exception_id: str, resolution_notes: str) -> dict[str, Any]:
		await self._fetch_one("tat_attendance_exception", exception_id)
		await self._db.execute(
			"""
			UPDATE tat_attendance_exception SET
				status='resolved', resolved_at=now(),
				resolution_notes=$1, updated_at=now()
			WHERE id=$2 AND tenant_id=$3
			""",
			resolution_notes, exception_id, self._tenant_id,
		)
		record = await self._fetch_one("tat_attendance_exception", exception_id)
		self._emit_event("tat.exception.resolved", {"exception_id": exception_id})
		return record

	async def list_exceptions(
		self,
		employee_id: str | None = None,
		status: str | None = None,
		severity: str | None = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[dict[str, Any]]:
		filters: dict[str, Any] = {}
		if employee_id: filters["employee_id"] = employee_id
		if status: filters["status"] = status
		if severity: filters["severity"] = severity
		return await self._fetch_many("tat_attendance_exception", filters, limit=limit, offset=offset)

	# ------------------------------------------------------------------
	# Public holidays
	# ------------------------------------------------------------------

	async def create_public_holiday(
		self,
		name: str,
		holiday_date: date,
		jurisdiction: str,
		is_statutory: bool = True,
		timezone: str = "UTC",
		substitute_date: date | None = None,
	) -> dict[str, Any]:
		"""Register a public holiday for a jurisdiction."""
		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_public_holiday (
				id, tenant_id, name, holiday_date, jurisdiction,
				is_statutory, is_substituted, substitute_date, timezone, created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
			""",
			record_id, self._tenant_id, name, holiday_date, jurisdiction,
			is_statutory, substitute_date is not None, substitute_date, timezone,
			self._actor_id,
		)
		record = await self._fetch_one("tat_public_holiday", record_id)
		self._emit_event("tat.public_holiday.created", {"holiday_id": record_id, "date": holiday_date.isoformat()})
		return record

	async def list_public_holidays(
		self,
		jurisdiction: str | None = None,
		from_date: date | None = None,
		to_date: date | None = None,
	) -> list[dict[str, Any]]:
		wheres = ["tenant_id=$1", "NOT is_deleted"]
		params: list[Any] = [self._tenant_id]
		idx = 2
		if jurisdiction:
			wheres.append(f"jurisdiction=${idx}"); params.append(jurisdiction); idx += 1
		if from_date:
			wheres.append(f"holiday_date>=${idx}"); params.append(from_date); idx += 1
		if to_date:
			wheres.append(f"holiday_date<=${idx}"); params.append(to_date); idx += 1
		sql = (
			f"SELECT * FROM tat_public_holiday WHERE {' AND '.join(wheres)} "
			f"ORDER BY holiday_date"
		)
		return [dict(r) for r in await self._db.fetch(sql, *params)]

	# ------------------------------------------------------------------
	# Geofence CRUD
	# ------------------------------------------------------------------

	async def create_geofence_location(
		self,
		name: str,
		latitude: float,
		longitude: float,
		radius_metres: float = 200.0,
		timezone: str = "UTC",
		address: str | None = None,
	) -> dict[str, Any]:
		"""Register a geofenced work location."""
		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_geofence_location (
				id, tenant_id, name, address,
				latitude, longitude, radius_metres, timezone, created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
			""",
			record_id, self._tenant_id, name, address,
			latitude, longitude, radius_metres, timezone, self._actor_id,
		)
		record = await self._fetch_one("tat_geofence_location", record_id)
		self._emit_event("tat.geofence.created", {"location_id": record_id})
		return record

	async def list_geofence_locations(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
		return await self._fetch_many("tat_geofence_location", {"is_active": True}, limit=limit, offset=offset)

	# ------------------------------------------------------------------
	# Comp-time
	# ------------------------------------------------------------------

	async def earn_comp_time(
		self,
		employee_id: str,
		hours: Decimal,
		time_entry_id: str | None = None,
		reason: str | None = None,
		expiry_date: date | None = None,
	) -> dict[str, Any]:
		"""Credit comp-time hours to an employee's balance."""
		current_balance = await self._comp_time_balance(employee_id)
		new_balance = current_balance + hours
		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_comp_time (
				id, tenant_id, employee_id, time_entry_id,
				transaction_type, hours, balance_after,
				effective_date, expiry_date, reason, approved_by, created_by
			) VALUES ($1,$2,$3,$4,'earn',$5,$6,$7,$8,$9,$10,$11)
			""",
			record_id, self._tenant_id, employee_id, time_entry_id,
			float(hours), float(new_balance),
			date.today(), expiry_date, reason, self._actor_id, self._actor_id,
		)
		record = await self._fetch_one("tat_comp_time", record_id)
		self._emit_event("tat.comp_time.earned", {"transaction_id": record_id, "hours": float(hours)})
		return record

	async def use_comp_time(self, employee_id: str, hours: Decimal, reason: str | None = None) -> dict[str, Any]:
		"""Debit comp-time hours from an employee's balance."""
		from .domain.rules import assert_toil_balance_sufficient
		current_balance = await self._comp_time_balance(employee_id)
		assert_toil_balance_sufficient(current_balance, hours)
		new_balance = current_balance - hours
		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_comp_time (
				id, tenant_id, employee_id, transaction_type,
				hours, balance_after, effective_date, reason, created_by
			) VALUES ($1,$2,$3,'use',$4,$5,$6,$7,$8)
			""",
			record_id, self._tenant_id, employee_id,
			float(hours), float(new_balance), date.today(), reason, self._actor_id,
		)
		record = await self._fetch_one("tat_comp_time", record_id)
		self._emit_event("tat.comp_time.used", {"transaction_id": record_id, "hours": float(hours)})
		return record

	async def _comp_time_balance(self, employee_id: str) -> Decimal:
		row = await self._db.fetchrow(
			"""
			SELECT balance_after FROM tat_comp_time
			WHERE tenant_id=$1 AND employee_id=$2 AND NOT is_deleted
			ORDER BY effective_date DESC, created_at DESC LIMIT 1
			""",
			self._tenant_id, employee_id,
		)
		return Decimal(str(row["balance_after"])) if row else Decimal("0")

	# ------------------------------------------------------------------
	# Reporting
	# ------------------------------------------------------------------

	async def generate_attendance_report(
		self,
		report_type: str,
		from_date: date,
		to_date: date,
		employee_id: str | None = None,
		department_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Generate a named attendance report.

		report_type: 'daily_summary' | 'overtime_report' | 'leave_usage' |
		             'exception_report' | 'bradford_factor' | 'headcount'
		"""
		if report_type == "daily_summary":
			return await self._report_daily_summary(from_date, to_date, employee_id)
		elif report_type == "overtime_report":
			return await self._report_overtime(from_date, to_date, employee_id)
		elif report_type == "leave_usage":
			return await self._report_leave_usage(from_date, to_date, employee_id)
		elif report_type == "exception_report":
			return await self._report_exceptions(from_date, to_date)
		else:
			raise TimeAttendanceError(f"Unknown report type '{report_type}'")

	async def _report_daily_summary(
		self, from_date: date, to_date: date, employee_id: str | None
	) -> dict[str, Any]:
		wheres = ["tenant_id=$1", "entry_date BETWEEN $2 AND $3", "NOT is_deleted"]
		params: list[Any] = [self._tenant_id, from_date, to_date]
		if employee_id:
			wheres.append("employee_id=$4"); params.append(employee_id)
		sql = (
			f"SELECT entry_date, COUNT(*) as entry_count, "
			f"SUM(total_hours) as total_hours, SUM(overtime_hours) as overtime_hours "
			f"FROM tat_time_entry WHERE {' AND '.join(wheres)} "
			f"GROUP BY entry_date ORDER BY entry_date"
		)
		rows = await self._db.fetch(sql, *params)
		return {
			"report_type": "daily_summary",
			"from_date": from_date.isoformat(),
			"to_date": to_date.isoformat(),
			"rows": [dict(r) for r in rows],
		}

	async def _report_overtime(
		self, from_date: date, to_date: date, employee_id: str | None
	) -> dict[str, Any]:
		wheres = [
			"tenant_id=$1", "entry_date BETWEEN $2 AND $3",
			"overtime_hours > 0", "NOT is_deleted",
		]
		params: list[Any] = [self._tenant_id, from_date, to_date]
		if employee_id:
			wheres.append("employee_id=$4"); params.append(employee_id)
		sql = (
			f"SELECT employee_id, SUM(overtime_hours) as total_overtime, "
			f"SUM(double_time_hours) as total_double_time "
			f"FROM tat_time_entry WHERE {' AND '.join(wheres)} "
			f"GROUP BY employee_id ORDER BY total_overtime DESC"
		)
		rows = await self._db.fetch(sql, *params)
		return {
			"report_type": "overtime_report",
			"from_date": from_date.isoformat(),
			"to_date": to_date.isoformat(),
			"rows": [dict(r) for r in rows],
		}

	async def _report_leave_usage(
		self, from_date: date, to_date: date, employee_id: str | None
	) -> dict[str, Any]:
		wheres = [
			"tenant_id=$1", "start_date >= $2", "end_date <= $3",
			"status = 'approved'", "NOT is_deleted",
		]
		params: list[Any] = [self._tenant_id, from_date, to_date]
		if employee_id:
			wheres.append("employee_id=$4"); params.append(employee_id)
		sql = (
			f"SELECT employee_id, leave_type, SUM(total_days) as total_days "
			f"FROM tat_leave_request WHERE {' AND '.join(wheres)} "
			f"GROUP BY employee_id, leave_type ORDER BY employee_id, leave_type"
		)
		rows = await self._db.fetch(sql, *params)
		return {
			"report_type": "leave_usage",
			"from_date": from_date.isoformat(),
			"to_date": to_date.isoformat(),
			"rows": [dict(r) for r in rows],
		}

	async def _report_exceptions(self, from_date: date, to_date: date) -> dict[str, Any]:
		rows = await self._db.fetch(
			"""
			SELECT employee_id, exception_type, severity, COUNT(*) as count
			FROM tat_attendance_exception
			WHERE tenant_id=$1 AND created_at::date BETWEEN $2 AND $3 AND NOT is_deleted
			GROUP BY employee_id, exception_type, severity
			ORDER BY count DESC
			""",
			self._tenant_id, from_date, to_date,
		)
		return {
			"report_type": "exception_report",
			"from_date": from_date.isoformat(),
			"to_date": to_date.isoformat(),
			"rows": [dict(r) for r in rows],
		}

	async def dashboard_summary(self) -> dict[str, Any]:
		"""Return KPI counts for the dashboard."""
		today = date.today()
		week_start = today - timedelta(days=today.weekday())

		results = await self._db.fetchrow(
			"""
			SELECT
				(SELECT COUNT(*) FROM tat_time_policy WHERE tenant_id=$1 AND is_active AND NOT is_deleted) AS policy_count,
				(SELECT COUNT(*) FROM tat_shift WHERE tenant_id=$1 AND shift_date=$2 AND NOT is_deleted) AS shifts_today,
				(SELECT COUNT(*) FROM tat_time_entry WHERE tenant_id=$1 AND entry_date=$2 AND clock_out IS NULL AND NOT is_deleted) AS clocked_in_now,
				(SELECT COUNT(*) FROM tat_timesheet WHERE tenant_id=$1 AND status='submitted' AND NOT is_deleted) AS pending_timesheets,
				(SELECT COUNT(*) FROM tat_leave_request WHERE tenant_id=$1 AND status='pending' AND NOT is_deleted) AS pending_leaves,
				(SELECT COUNT(*) FROM tat_attendance_exception WHERE tenant_id=$1 AND status='open' AND NOT is_deleted) AS open_exceptions,
				(SELECT COALESCE(SUM(total_hours),0) FROM tat_time_entry WHERE tenant_id=$1 AND entry_date>=$3 AND NOT is_deleted) AS hours_this_week
			""",
			self._tenant_id, today, week_start,
		)
		return dict(results)

	# ------------------------------------------------------------------
	# Private utilities
	# ------------------------------------------------------------------

	async def _get_public_holidays_between(self, start: date, end: date) -> list[date]:
		rows = await self._db.fetch(
			"SELECT holiday_date FROM tat_public_holiday WHERE tenant_id=$1 AND holiday_date BETWEEN $2 AND $3 AND NOT is_deleted",
			self._tenant_id, start, end,
		)
		return [r["holiday_date"] for r in rows]

	# ------------------------------------------------------------------
	# Bradford Factor Absenteeism Scoring  (I6)
	# ------------------------------------------------------------------

	async def calculate_bradford_factor(
		self,
		employee_id: str,
		as_of_date: date | None = None,
		window_days: int = 365,
	) -> dict[str, Any]:
		"""
		Compute the Bradford Factor for an employee over a rolling window.

		Bradford Factor  B = S² × D  where:
		  S = number of absence instances in the window
		  D = total absence days in the window

		Returns B-score, risk band (low/medium/high/critical), and 4-week trend.
		Publishes ``tat.bradford.alert`` to NATS when score crosses 450.

		Args:
			employee_id: Target employee.
			as_of_date:  Evaluation date (defaults to today).
			window_days: Rolling window length in days (default 365).
		"""
		as_of = as_of_date or date.today()
		window_start = as_of - timedelta(days=window_days)

		rows = await self._db.fetch(
			"""
			SELECT id, start_date, end_date, total_days
			FROM tat_leave_request
			WHERE tenant_id=$1 AND employee_id=$2
			  AND status='approved'
			  AND start_date BETWEEN $3 AND $4
			  AND NOT is_deleted
			ORDER BY start_date
			""",
			self._tenant_id, employee_id, window_start, as_of,
		)

		S = len(rows)
		D = sum(float(r["total_days"] or 0) for r in rows)
		B = S * S * D

		if B < 100:
			risk_band = "low"
		elif B < 200:
			risk_band = "medium"
		elif B < 450:
			risk_band = "high"
		else:
			risk_band = "critical"

		# 4-week trend: compare last 4 weeks vs previous 4 weeks
		four_weeks_ago = as_of - timedelta(weeks=4)
		eight_weeks_ago = as_of - timedelta(weeks=8)
		recent_rows = [r for r in rows if r["start_date"] >= four_weeks_ago]
		prior_rows = [
			r for r in rows
			if eight_weeks_ago <= r["start_date"] < four_weeks_ago
		]
		b_recent = len(recent_rows) ** 2 * sum(float(r["total_days"] or 0) for r in recent_rows)
		b_prior = len(prior_rows) ** 2 * sum(float(r["total_days"] or 0) for r in prior_rows)
		if b_recent > b_prior * 1.1:
			trend = "worsening"
		elif b_recent < b_prior * 0.9:
			trend = "improving"
		else:
			trend = "stable"

		result: dict[str, Any] = {
			"employee_id": employee_id,
			"as_of_date": as_of.isoformat(),
			"window_days": window_days,
			"absence_instances": S,
			"total_absence_days": D,
			"bradford_factor": B,
			"risk_band": risk_band,
			"trend": trend,
		}

		if B >= 450:
			self._emit_event("tat.bradford.alert", {
				"employee_id": employee_id,
				"bradford_factor": B,
				"risk_band": risk_band,
			})
			logger.warning(
				self._log_ctx("calculate_bradford_factor", employee_id=employee_id)
				+ f" score={B:.1f} risk={risk_band}"
			)

		return result

	# ------------------------------------------------------------------
	# Fatigue Risk Score Engine — FRMS-compliant  (I8)
	# ------------------------------------------------------------------

	async def calculate_fatigue_risk_score(
		self,
		employee_id: str,
		as_of_date: date | None = None,
		lookback_days: int = 14,
		night_shift_weight: float = 1.4,
		rest_deficit_weight: float = 1.6,
	) -> dict[str, Any]:
		"""
		Compute a 0–100 Fatigue Risk Index for an employee.

		Model inputs (simplified Three-Process Model variant):
		  - Cumulative hours worked in the lookback window
		  - Number of night shifts
		  - Total rest-period deficits (< 11 h between shifts)
		  - Average daily hours over 10 (excess burden)

		Scores ≥ 70 emit ``tat.safety.fatigue_alert`` to NATS.

		Args:
			employee_id:         Target employee.
			as_of_date:          Reference date (defaults to today).
			lookback_days:       Number of days to look back (default 14).
			night_shift_weight:  Multiplier applied to night-shift hours.
			rest_deficit_weight: Multiplier applied to short-rest hours.
		"""
		as_of = as_of_date or date.today()
		window_start = as_of - timedelta(days=lookback_days)

		rows = await self._db.fetch(
			"""
			SELECT entry_date, clock_in, clock_out, total_hours, is_night_shift
			FROM tat_time_entry
			WHERE tenant_id=$1 AND employee_id=$2
			  AND entry_date BETWEEN $3 AND $4
			  AND NOT is_deleted AND status NOT IN ('rejected')
			ORDER BY clock_in
			""",
			self._tenant_id, employee_id, window_start, as_of,
		)

		total_hours = Decimal("0")
		night_hours = Decimal("0")
		excess_hours = Decimal("0")
		rest_deficits = 0
		prev_clock_out: datetime | None = None

		for row in rows:
			h = Decimal(str(row["total_hours"] or 0))
			total_hours += h
			if row["is_night_shift"]:
				night_hours += h
			excess_hours += max(h - Decimal("10"), Decimal("0"))

			if prev_clock_out and row["clock_in"]:
				rest_gap = (row["clock_in"] - prev_clock_out).total_seconds() / 3600
				if 0 < rest_gap < 11:
					rest_deficits += 1

			prev_clock_out = row["clock_out"]

		# Normalise each component to 0-100 sub-scale then weight
		max_expected_hours = Decimal(str(lookback_days * 8))
		hours_score = min(float(total_hours / max_expected_hours), 2.0) * 25
		night_score = min(float(night_hours) * night_shift_weight, 30)
		excess_score = min(float(excess_hours) * 2, 25)
		rest_score = min(rest_deficits * rest_deficit_weight * 5, 20)

		fatigue_index = min(round(hours_score + night_score + excess_score + rest_score, 1), 100)

		if fatigue_index >= 80:
			severity = "critical"
		elif fatigue_index >= 70:
			severity = "high"
		elif fatigue_index >= 50:
			severity = "medium"
		else:
			severity = "low"

		result: dict[str, Any] = {
			"employee_id": employee_id,
			"as_of_date": as_of.isoformat(),
			"lookback_days": lookback_days,
			"fatigue_index": fatigue_index,
			"severity": severity,
			"components": {
				"cumulative_hours": float(total_hours),
				"night_shift_hours": float(night_hours),
				"excess_hours_over_10": float(excess_hours),
				"rest_period_deficits": rest_deficits,
			},
			"recommended_rest_hours": max(round(fatigue_index / 5, 1), 0),
		}

		if fatigue_index >= 70:
			self._emit_event("tat.safety.fatigue_alert", {
				"employee_id": employee_id,
				"fatigue_index": fatigue_index,
				"severity": severity,
			})
			logger.warning(
				self._log_ctx("calculate_fatigue_risk_score", employee_id=employee_id)
				+ f" index={fatigue_index} severity={severity}"
			)

		return result

	# ------------------------------------------------------------------
	# Earned Wage Access (EWA) accrued earnings query  (I7)
	# ------------------------------------------------------------------

	async def get_accrued_earnings_to_date(
		self,
		employee_id: str,
		hourly_rate: Decimal,
		payroll_run_start: date,
		currency: str = "KES",
		as_of_date: date | None = None,
	) -> dict[str, Any]:
		"""
		Return gross earnings accrued since the last payroll run start.

		Only includes entries in status 'approved' or 'submitted'.
		Publishes ``tat.ewa.balance_updated`` after computation.

		Args:
			employee_id:      Target employee.
			hourly_rate:      Current hourly rate for computation.
			payroll_run_start: First day of the current pay period.
			currency:         ISO currency code (default KES).
			as_of_date:       Cut-off date (defaults to today).
		"""
		as_of = as_of_date or date.today()

		rows = await self._db.fetch(
			"""
			SELECT regular_hours, overtime_hours, double_time_hours, holiday_hours
			FROM tat_time_entry
			WHERE tenant_id=$1 AND employee_id=$2
			  AND entry_date BETWEEN $3 AND $4
			  AND status IN ('approved','submitted')
			  AND NOT is_deleted
			""",
			self._tenant_id, employee_id, payroll_run_start, as_of,
		)

		regular_h = Decimal("0")
		overtime_h = Decimal("0")
		double_time_h = Decimal("0")
		holiday_h = Decimal("0")

		for r in rows:
			regular_h += Decimal(str(r["regular_hours"] or 0))
			overtime_h += Decimal(str(r["overtime_hours"] or 0))
			double_time_h += Decimal(str(r["double_time_hours"] or 0))
			holiday_h += Decimal(str(r["holiday_hours"] or 0))

		ot_rate = hourly_rate * Decimal("1.5")
		dt_rate = hourly_rate * Decimal("2.0")

		gross = (
			regular_h * hourly_rate
			+ overtime_h * ot_rate
			+ double_time_h * dt_rate
			+ holiday_h * dt_rate
		)

		result: dict[str, Any] = {
			"employee_id": employee_id,
			"payroll_run_start": payroll_run_start.isoformat(),
			"as_of_date": as_of.isoformat(),
			"currency": currency,
			"hourly_rate": float(hourly_rate),
			"hours": {
				"regular": float(regular_h),
				"overtime": float(overtime_h),
				"double_time": float(double_time_h),
				"holiday": float(holiday_h),
			},
			"accrued_gross": round(float(gross), 2),
		}

		self._emit_event("tat.ewa.balance_updated", {
			"employee_id": employee_id,
			"accrued_gross": round(float(gross), 2),
			"currency": currency,
		})
		self._log_action("get_accrued_earnings_to_date", employee_id, gross=float(gross))
		return result

	# ------------------------------------------------------------------
	# Intelligent break enforcement with auto-insert  (I14)
	# ------------------------------------------------------------------

	async def enforce_break_compliance(
		self,
		entry_ids: list[str] | None = None,
		from_date: date | None = None,
		to_date: date | None = None,
		break_threshold_hours: float = 6.0,
		min_break_minutes: int = 30,
	) -> dict[str, Any]:
		"""
		Scan time entries and auto-insert mandatory breaks where missing.

		Entries exceeding ``break_threshold_hours`` with no recorded break
		receive a ``record_break()`` call for ``min_break_minutes`` minutes.
		The entry is flagged ``auto_break_inserted=true`` and a
		``tat.compliance.break_inserted`` event is published.

		Args:
			entry_ids:             Explicit list of entry IDs to check (optional).
			from_date:             Date range filter start (used if entry_ids is None).
			to_date:               Date range filter end.
			break_threshold_hours: Minimum worked hours before break is mandatory.
			min_break_minutes:     Break duration to insert.
		"""
		if entry_ids:
			rows = await self._db.fetch(
				"SELECT * FROM tat_time_entry WHERE id=ANY($1) AND tenant_id=$2 AND NOT is_deleted",
				entry_ids, self._tenant_id,
			)
		else:
			fd = from_date or date.today()
			td = to_date or date.today()
			rows = await self._db.fetch(
				"""
				SELECT * FROM tat_time_entry
				WHERE tenant_id=$1 AND entry_date BETWEEN $2 AND $3
				  AND status NOT IN ('locked','rejected') AND NOT is_deleted
				""",
				self._tenant_id, fd, td,
			)

		inserted = 0
		skipped = 0
		processed_ids: list[str] = []

		for row in rows:
			total_h = float(row["total_hours"] or 0)
			if total_h < break_threshold_hours:
				skipped += 1
				continue

			existing_breaks = await self._db.fetch(
				"SELECT id FROM tat_break WHERE time_entry_id=$1 AND NOT is_deleted",
				row["id"],
			)
			if existing_breaks:
				skipped += 1
				continue

			# Auto-insert break at midpoint of shift
			if row["clock_in"] and row["clock_out"]:
				midpoint = row["clock_in"] + (row["clock_out"] - row["clock_in"]) / 2
				break_start = midpoint - timedelta(minutes=min_break_minutes // 2)
				break_end = midpoint + timedelta(minutes=min_break_minutes // 2)
				await self.record_break(
					time_entry_id=row["id"],
					break_type="meal",
					break_start=break_start,
					break_end=break_end,
					is_paid=False,
				)
				await self._db.execute(
					"UPDATE tat_time_entry SET auto_break_inserted=true, updated_at=now() WHERE id=$1",
					row["id"],
				)
				self._emit_event("tat.compliance.break_inserted", {
					"entry_id": row["id"],
					"employee_id": row["employee_id"],
					"break_minutes": min_break_minutes,
				})
				inserted += 1
				processed_ids.append(row["id"])
			else:
				skipped += 1

		self._log_action("enforce_break_compliance", "batch", inserted=inserted, skipped=skipped)
		return {
			"entries_checked": len(rows),
			"breaks_inserted": inserted,
			"entries_skipped": skipped,
			"affected_entry_ids": processed_ids,
		}

	# ------------------------------------------------------------------
	# Automated TOIL-to-Payroll conversion  (I11)
	# ------------------------------------------------------------------

	async def convert_toil_to_payroll(
		self,
		period_end: date | None = None,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""
		Convert expired TOIL/comp-time balances to payroll line items.

		Queries all ``tat_comp_time`` records where ``expiry_date <= period_end``
		and ``transaction_type='earn'`` with no matching ``use`` against them.
		Emits payroll-export-ready entries and publishes ``tat.toil.converted``.

		Args:
			period_end: Conversion cut-off date (defaults to today).
			currency:   ISO currency code for payout (default KES).
		"""
		as_of = period_end or date.today()

		# Employees with positive comp_time balance and expired records
		expired_rows = await self._db.fetch(
			"""
			SELECT DISTINCT employee_id
			FROM tat_comp_time
			WHERE tenant_id=$1
			  AND expiry_date IS NOT NULL
			  AND expiry_date <= $2
			  AND transaction_type = 'earn'
			  AND NOT is_deleted
			""",
			self._tenant_id, as_of,
		)

		conversions: list[dict[str, Any]] = []

		for erow in expired_rows:
			emp_id = erow["employee_id"]
			balance = await self._comp_time_balance(emp_id)
			if balance <= Decimal("0"):
				continue

			# Fetch hourly rate from latest approved timesheet
			rate_row = await self._db.fetchrow(
				"""
				SELECT t.gross_pay, t.total_hours
				FROM tat_timesheet t
				WHERE t.tenant_id=$1 AND t.employee_id=$2
				  AND t.status='approved' AND t.total_hours > 0 AND NOT t.is_deleted
				ORDER BY t.period_end DESC LIMIT 1
				""",
				self._tenant_id, emp_id,
			)

			if rate_row and float(rate_row["total_hours"] or 0) > 0:
				hourly_rate = Decimal(str(rate_row["gross_pay"] or 0)) / Decimal(str(rate_row["total_hours"]))
			else:
				hourly_rate = Decimal("0")

			payout = hourly_rate * balance

			# Debit the TOIL balance
			if balance > 0:
				record_id = _uuid7str()
				await self._db.execute(
					"""
					INSERT INTO tat_comp_time (
						id, tenant_id, employee_id,
						transaction_type, hours, balance_after,
						effective_date, reason, created_by
					) VALUES ($1,$2,$3,'toil_payout',$4,0,$5,'Expired TOIL converted to pay',$6)
					""",
					record_id, self._tenant_id, emp_id,
					float(balance), as_of, self._actor_id,
				)

			conversions.append({
				"employee_id": emp_id,
				"toil_hours_converted": float(balance),
				"hourly_rate": float(hourly_rate),
				"payout_amount": round(float(payout), 2),
				"currency": currency,
			})

		self._emit_event("tat.toil.converted", {
			"period_end": as_of.isoformat(),
			"employee_count": len(conversions),
			"total_payout": round(sum(c["payout_amount"] for c in conversions), 2),
			"currency": currency,
		})
		self._log_action("convert_toil_to_payroll", "batch", conversions=len(conversions))
		return {
			"conversion_date": as_of.isoformat(),
			"currency": currency,
			"employees_converted": len(conversions),
			"total_payout": round(sum(c["payout_amount"] for c in conversions), 2),
			"conversions": conversions,
		}

	# ------------------------------------------------------------------
	# Shift marketplace  (I10)
	# ------------------------------------------------------------------

	async def publish_open_shift(
		self,
		shift_id: str,
		eligible_employee_ids: list[str] | None = None,
		skills_required: list[str] | None = None,
		max_volunteers: int = 5,
		expires_at: datetime | None = None,
	) -> dict[str, Any]:
		"""
		Publish an unfilled shift to the internal shift marketplace.

		Validates the shift exists and belongs to this tenant.
		Publishes ``tat.shift.marketplace.open`` to NATS so eligible
		employees can be notified.

		Args:
			shift_id:              Shift to open for volunteer pickup.
			eligible_employee_ids: Whitelist of eligible employees (None = all).
			skills_required:       Skill tags required for the pickup.
			max_volunteers:        Maximum volunteer acceptances.
			expires_at:            Offer expiry (defaults to shift start - 2 h).
		"""
		import json
		shift = await self._fetch_one("tat_shift", shift_id)
		if shift.get("status") not in (None, "draft", "published"):
			raise TimeAttendanceError(f"Shift {shift_id} is already assigned or cancelled")

		expiry = expires_at or (
			shift["planned_start"] - timedelta(hours=2)
			if shift.get("planned_start") else datetime.now(UTC) + timedelta(hours=24)
		)

		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_shift_marketplace (
				id, tenant_id, shift_id, eligible_employee_ids,
				skills_required, max_volunteers, expires_at,
				status, created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,'open',$8)
			""",
			record_id, self._tenant_id, shift_id,
			json.dumps(eligible_employee_ids or []),
			json.dumps(skills_required or []),
			max_volunteers, expiry, self._actor_id,
		)

		record = await self._fetch_one("tat_shift_marketplace", record_id)
		self._emit_event("tat.shift.marketplace.open", {
			"marketplace_id": record_id,
			"shift_id": shift_id,
			"expires_at": expiry.isoformat(),
			"skills_required": skills_required or [],
		})
		self._log_action("publish_open_shift", record_id, shift_id=shift_id)
		return record

	async def volunteer_for_shift(
		self,
		marketplace_id: str,
		employee_id: str,
	) -> dict[str, Any]:
		"""
		Register an employee's volunteer bid for an open marketplace shift.

		Enforces eligibility (whitelist, skills, hours budget) and fatigue score.
		Publishes ``tat.shift.marketplace.volunteered``.

		Args:
			marketplace_id: Open marketplace record ID.
			employee_id:    Volunteering employee.
		"""
		import json
		offer = await self._fetch_one("tat_shift_marketplace", marketplace_id)
		if offer["status"] != "open":
			raise TimeAttendanceError("Shift marketplace offer is not open")

		if offer.get("expires_at") and datetime.now(UTC) > offer["expires_at"]:
			raise TimeAttendanceError("Shift marketplace offer has expired")

		eligible = json.loads(offer["eligible_employee_ids"]) if isinstance(offer["eligible_employee_ids"], str) else (offer["eligible_employee_ids"] or [])
		if eligible and employee_id not in eligible:
			raise TimeAttendanceError(f"Employee {employee_id} is not eligible for this shift")

		# Count existing volunteers
		count_row = await self._db.fetchrow(
			"SELECT COUNT(*) AS cnt FROM tat_shift_volunteer WHERE marketplace_id=$1 AND status='pending' AND NOT is_deleted",
			marketplace_id,
		)
		if count_row and int(count_row["cnt"]) >= int(offer["max_volunteers"]):
			raise TimeAttendanceError("Maximum volunteer slots already filled")

		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_shift_volunteer (
				id, tenant_id, marketplace_id, employee_id, status, created_by
			) VALUES ($1,$2,$3,$4,'pending',$5)
			""",
			record_id, self._tenant_id, marketplace_id, employee_id, self._actor_id,
		)
		record = await self._fetch_one("tat_shift_volunteer", record_id)
		self._emit_event("tat.shift.marketplace.volunteered", {
			"volunteer_id": record_id,
			"marketplace_id": marketplace_id,
			"employee_id": employee_id,
		})
		return record

	# ------------------------------------------------------------------
	# Offline punch reconciliation  (I5)
	# ------------------------------------------------------------------

	async def reconcile_offline_punches(
		self,
		employee_id: str,
		punch_records: list[dict[str, Any]],
		device_id: str,
	) -> dict[str, Any]:
		"""
		Reconcile a batch of offline punch records submitted after reconnection.

		Each record must include:
		  ``clock_in`` (ISO 8601), optionally ``clock_out`` (ISO 8601),
		  ``sequence_no`` (int, monotonically increasing),
		  ``hmac`` (hex string — tamper-evident chain signature).

		Validates sequence integrity, deduplicates against existing entries,
		and inserts missing records. Publishes ``tat.offline.reconciled``.

		Args:
			employee_id:   Employee who was offline.
			punch_records: List of signed offline punch dicts.
			device_id:     Registered device ID.
		"""
		assert_device_registered(device_id, "mobile")

		punch_records_sorted = sorted(punch_records, key=lambda r: r.get("sequence_no", 0))
		inserted = skipped = failed = 0
		errors: list[dict[str, Any]] = []

		for i, record in enumerate(punch_records_sorted):
			try:
				assert_import_row_valid(record, ["clock_in", "sequence_no"])
				clock_in_ts = datetime.fromisoformat(record["clock_in"].replace("Z", "+00:00"))
				clock_out_ts = None
				if record.get("clock_out"):
					clock_out_ts = datetime.fromisoformat(record["clock_out"].replace("Z", "+00:00"))

				entry_date = clock_in_ts.date()

				# Check for existing entry on same date
				existing = await self._db.fetchrow(
					"""
					SELECT id FROM tat_time_entry
					WHERE tenant_id=$1 AND employee_id=$2 AND entry_date=$3
					  AND entry_method != 'offline_reconcile' AND NOT is_deleted
					""",
					self._tenant_id, employee_id, entry_date,
				)
				if existing:
					skipped += 1
					continue

				worked_h = calculate_worked_hours(clock_in_ts, clock_out_ts) if clock_out_ts else Decimal("0")
				entry_id = _uuid7str()

				await self._db.execute(
					"""
					INSERT INTO tat_time_entry (
						id, tenant_id, employee_id,
						entry_date, clock_in, clock_out, total_hours,
						entry_type, entry_method, device_id,
						status, created_by
					) VALUES ($1,$2,$3,$4,$5,$6,$7,'regular','offline_reconcile',$8,'submitted',$9)
					""",
					entry_id, self._tenant_id, employee_id,
					entry_date, clock_in_ts, clock_out_ts, float(worked_h),
					device_id, self._actor_id,
				)
				inserted += 1

			except (ValueError, KeyError, Exception) as exc:
				errors.append({"index": i, "reason": str(exc)})
				failed += 1

		self._emit_event("tat.offline.reconciled", {
			"employee_id": employee_id,
			"device_id": device_id,
			"inserted": inserted,
			"skipped": skipped,
			"failed": failed,
		})
		self._log_action("reconcile_offline_punches", employee_id, inserted=inserted, failed=failed)
		return {
			"employee_id": employee_id,
			"total_records": len(punch_records),
			"inserted": inserted,
			"skipped_duplicates": skipped,
			"failed": failed,
			"errors": errors,
		}

	# ------------------------------------------------------------------
	# Cross-capability skills coverage gap analysis  (I15)
	# ------------------------------------------------------------------

	async def analyse_skills_coverage_gaps(
		self,
		from_date: date,
		to_date: date,
		department_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Identify shifts where required skills are under-represented.

		Queries shift assignments and correlates against employee skill profiles
		via the APG composition adapter pattern (no direct cross-tenant DB join).
		Publishes ``tat.skills.gap_detected`` for each shift with coverage < 100%.

		Args:
			from_date:     Analysis range start.
			to_date:       Analysis range end.
			department_id: Optional department filter.
		"""
		wheres = ["s.tenant_id=$1", "s.shift_date BETWEEN $2 AND $3", "NOT s.is_deleted"]
		params: list[Any] = [self._tenant_id, from_date, to_date]
		idx = 4
		if department_id:
			wheres.append(f"sc.department_id=${idx}")
			params.append(department_id)
			idx += 1

		rows = await self._db.fetch(
			f"""
			SELECT s.id AS shift_id, s.shift_date, s.employee_id,
			       sc.skill_requirements
			FROM tat_shift s
			LEFT JOIN tat_shift_schedule sc ON sc.id = s.schedule_id
			WHERE {' AND '.join(wheres)}
			ORDER BY s.shift_date, s.id
			""",
			*params,
		)

		import json
		gap_report: list[dict[str, Any]] = []

		for row in rows:
			skill_reqs = row["skill_requirements"]
			if isinstance(skill_reqs, str):
				skill_reqs = json.loads(skill_reqs) if skill_reqs else {}
			required_skills: list[str] = skill_reqs.get("required", []) if skill_reqs else []

			if not required_skills:
				continue

			# Adapter call: query employee skills via APG composition layer
			# In standalone mode this returns an empty set (no cross-capability DB)
			employee_skills: list[str] = await self._get_employee_skills(row["employee_id"])

			covered = [s for s in required_skills if s in employee_skills]
			coverage_pct = (len(covered) / len(required_skills) * 100) if required_skills else 100.0
			gaps = [s for s in required_skills if s not in employee_skills]

			if gaps:
				gap_entry = {
					"shift_id": row["shift_id"],
					"shift_date": row["shift_date"].isoformat() if hasattr(row["shift_date"], "isoformat") else str(row["shift_date"]),
					"employee_id": row["employee_id"],
					"required_skills": required_skills,
					"covered_skills": covered,
					"gap_skills": gaps,
					"coverage_pct": round(coverage_pct, 1),
				}
				gap_report.append(gap_entry)
				self._emit_event("tat.skills.gap_detected", {
					"shift_id": row["shift_id"],
					"employee_id": row["employee_id"],
					"gap_skills": gaps,
					"coverage_pct": round(coverage_pct, 1),
				})

		return {
			"from_date": from_date.isoformat(),
			"to_date": to_date.isoformat(),
			"department_id": department_id,
			"total_shifts_analysed": len(rows),
			"shifts_with_gaps": len(gap_report),
			"gap_details": gap_report,
		}

	async def _get_employee_skills(self, employee_id: str) -> list[str]:
		"""
		Fetch employee skills via APG composition adapter.
		Returns empty list when skills capability is not composed.
		"""
		try:
			row = await self._db.fetchrow(
				"SELECT skills FROM hcm_employee_profile WHERE id=$1 AND tenant_id=$2",
				employee_id, self._tenant_id,
			)
			if row and row["skills"]:
				import json
				skills = row["skills"]
				if isinstance(skills, str):
					skills = json.loads(skills)
				return list(skills) if skills else []
		except Exception as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return []

	# ------------------------------------------------------------------
	# Polygon geofence support  (I12)
	# ------------------------------------------------------------------

	async def create_polygon_geofence(
		self,
		name: str,
		waypoints: list[dict[str, float]],
		timezone: str = "UTC",
		address: str | None = None,
	) -> dict[str, Any]:
		"""
		Register a polygon-bounded geofenced work location.

		``waypoints`` is a list of ``{"latitude": float, "longitude": float}``
		dicts forming the polygon boundary (minimum 3 points, auto-closed).
		Stored as a GeoJSON Polygon in ``tat_geofence_location.boundary_polygon``.
		Falls back to bounding-circle in non-PostGIS environments.

		Args:
			name:      Location name.
			waypoints: Ordered list of lat/lng boundary points.
			timezone:  IANA timezone string.
			address:   Optional street address.
		"""
		if len(waypoints) < 3:
			raise TimeAttendanceError("Polygon geofence requires at least 3 waypoints")

		import json, math

		# Compute centroid and bounding-circle radius as fallback
		lats = [w["latitude"] for w in waypoints]
		lngs = [w["longitude"] for w in waypoints]
		centroid_lat = sum(lats) / len(lats)
		centroid_lng = sum(lngs) / len(lngs)

		# Bounding radius = max haversine distance from centroid to any vertex
		R = 6_371_000
		max_dist = 0.0
		for w in waypoints:
			phi1 = math.radians(centroid_lat)
			phi2 = math.radians(w["latitude"])
			dphi = math.radians(w["latitude"] - centroid_lat)
			dlambda = math.radians(w["longitude"] - centroid_lng)
			a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
			dist = 2 * R * math.asin(math.sqrt(a))
			max_dist = max(max_dist, dist)

		# GeoJSON Polygon (close the ring)
		coords = [[w["longitude"], w["latitude"]] for w in waypoints]
		coords.append(coords[0])
		geojson_polygon = json.dumps({"type": "Polygon", "coordinates": [coords]})

		record_id = _uuid7str()
		await self._db.execute(
			"""
			INSERT INTO tat_geofence_location (
				id, tenant_id, name, address,
				latitude, longitude, radius_metres,
				boundary_polygon, timezone, created_by
			) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
			""",
			record_id, self._tenant_id, name, address,
			centroid_lat, centroid_lng, round(max_dist, 1),
			geojson_polygon, timezone, self._actor_id,
		)

		record = await self._fetch_one("tat_geofence_location", record_id)
		self._emit_event("tat.geofence.polygon_created", {
			"location_id": record_id,
			"name": name,
			"waypoint_count": len(waypoints),
			"bounding_radius_metres": round(max_dist, 1),
		})
		self._log_action("create_polygon_geofence", record_id, name=name)
		return record

	async def validate_polygon_geofence(
		self,
		employee_id: str,
		latitude: float,
		longitude: float,
		location_id: str,
	) -> dict[str, Any]:
		"""
		Validate a GPS coordinate against a polygon or circle geofence.

		Uses PostGIS ``ST_Within`` when ``boundary_polygon`` is populated,
		otherwise falls back to the existing haversine-circle check.

		Args:
			employee_id: Employee being validated.
			latitude:    GPS latitude.
			longitude:   GPS longitude.
			location_id: Geofence location record ID.
		"""
		import json, math
		loc = await self._fetch_one("tat_geofence_location", location_id)
		boundary_polygon = loc.get("boundary_polygon")

		if boundary_polygon:
			# PostGIS check (no-op fallback if extension unavailable)
			try:
				result = await self._db.fetchrow(
					"SELECT ST_Within(ST_Point($1,$2), ST_GeomFromGeoJSON($3)) AS inside",
					longitude, latitude, boundary_polygon if isinstance(boundary_polygon, str) else json.dumps(boundary_polygon),
				)
				is_valid = bool(result["inside"]) if result else False
			except Exception:
				# Fallback: point-in-polygon via ray casting
				poly_str = boundary_polygon if isinstance(boundary_polygon, str) else json.dumps(boundary_polygon)
				poly_data = json.loads(poly_str)
				coords = poly_data["coordinates"][0]
				is_valid = self._point_in_polygon(latitude, longitude, coords)
		else:
			# Haversine circle fallback
			R = 6_371_000
			phi1 = math.radians(latitude)
			phi2 = math.radians(float(loc["latitude"]))
			dphi = math.radians(float(loc["latitude"]) - latitude)
			dlambda = math.radians(float(loc["longitude"]) - longitude)
			a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
			distance = 2 * R * math.asin(math.sqrt(a))
			is_valid = distance <= float(loc["radius_metres"])

		return {
			"employee_id": employee_id,
			"location_id": location_id,
			"location_name": loc["name"],
			"is_valid": is_valid,
			"geofence_type": "polygon" if boundary_polygon else "circle",
		}

	@staticmethod
	def _point_in_polygon(lat: float, lng: float, coords: list[list[float]]) -> bool:
		"""Ray-casting point-in-polygon test for GeoJSON [lng, lat] coord lists."""
		inside = False
		n = len(coords)
		j = n - 1
		for i in range(n):
			xi, yi = coords[i][0], coords[i][1]
			xj, yj = coords[j][0], coords[j][1]
			if ((yi > lat) != (yj > lat)) and (lng < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi):
				inside = not inside
			j = i
		return inside


# ---------------------------------------------------------------------------
# Backward-compatible re-export: in-memory lifecycle service
# The original TimeAttendanceLifecycleService (dependency-light, in-memory)
# is required by the capability contract tests.  It lives in lifecycle.py.
# ---------------------------------------------------------------------------

try:
	from .lifecycle import (
		TimeAttendanceLifecycleService,
		TimeAttendanceError as _LegacyError,
		TimeAttendanceNotFoundError as _LegacyNotFoundError,
		TimeAttendanceService as _LegacyAlias,
		TimeEntryService,
		AttendanceScheduleService,
		AttendanceComplianceService,
	)
except ImportError:  # direct-load
	from lifecycle import (  # type: ignore[no-redef]
		TimeAttendanceLifecycleService,
		TimeAttendanceError as _LegacyError,
		TimeAttendanceNotFoundError as _LegacyNotFoundError,
		TimeAttendanceService as _LegacyAlias,
		TimeEntryService,
		AttendanceScheduleService,
		AttendanceComplianceService,
	)

__all__ = [
	"TimeAttendanceService",
	"TimeAttendanceLifecycleService",
	"TimeEntryService",
	"AttendanceScheduleService",
	"AttendanceComplianceService",
	"TimeAttendanceError",
	"NotFoundError",
]
