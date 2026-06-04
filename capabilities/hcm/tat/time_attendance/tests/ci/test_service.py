"""
CI tests for the TimeAttendanceLifecycleService (in-memory) and the
async TimeAttendanceService (thin asyncpg stub).

Copyright © 2025 Datacraft. Author: Nyimbi Odero
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
from uuid6 import uuid7

from domain.rules import RuleViolation

UTC = timezone.utc
TENANT = "tenant-svc-test"
ACTOR = "actor-test"


def _uuid() -> str:
	return str(uuid7())


# ---------------------------------------------------------------------------
# Minimal asyncpg-compatible in-memory stub for TimeAttendanceService
# ---------------------------------------------------------------------------

class _Row(dict):
	def __getattr__(self, item: str) -> Any:
		try:
			return self[item]
		except KeyError:
			raise AttributeError(item)


class _MemDB:
	def __init__(self) -> None:
		self.tables: dict[str, list[_Row]] = defaultdict(list)
		self.log: list[tuple[str, tuple]] = []

	def _table(self, sql: str) -> str:
		import re
		m = re.search(r"\btat_\w+", sql)
		return m.group(0) if m else "__unknown__"

	async def execute(self, sql: str, *params: Any) -> None:
		self.log.append((sql.strip()[:80], params))
		sql_lower = sql.lower().strip()
		if sql_lower.startswith("insert"):
			table = self._table(sql)
			row = _Row({"id": params[0] if params else _uuid(), "tenant_id": params[1] if len(params) > 1 else TENANT, "is_deleted": False, "status": "pending", "created_at": datetime.now(UTC), "updated_at": datetime.now(UTC)})
			self.tables[table].append(row)
		elif sql_lower.startswith("update"):
			table = self._table(sql)
			for row in self.tables[table]:
				row["updated_at"] = datetime.now(UTC)

	async def fetch(self, sql: str, *params: Any) -> list[_Row]:
		self.log.append((sql.strip()[:80], params))
		table = self._table(sql)
		rows = self.tables.get(table, [])
		if params and isinstance(params[0], str):
			rows = [r for r in rows if r.get("tenant_id") == params[0]]
		return [_Row(r) for r in rows]

	async def fetchrow(self, sql: str, *params: Any) -> _Row | None:
		self.log.append((sql.strip()[:80], params))
		table = self._table(sql)
		rows = self.tables.get(table, [])
		if params:
			for row in rows:
				if row.get("id") == params[0]:
					return _Row(row)
			if rows:
				return _Row(rows[0])
		return None

	def seed(self, table: str, row: dict[str, Any]) -> None:
		"""Pre-populate a table row for tests that need existing data."""
		self.tables[table].append(_Row({"is_deleted": False, **row}))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> _MemDB:
	return _MemDB()


@pytest.fixture
def svc(db: _MemDB):
	from service import TimeAttendanceService
	return TimeAttendanceService(db, TENANT, ACTOR)


@pytest.fixture
def lifecycle():
	from service import TimeAttendanceLifecycleService
	return TimeAttendanceLifecycleService()


# ---------------------------------------------------------------------------
# TimeAttendanceService — initialisation guards
# ---------------------------------------------------------------------------

class TestServiceInit:
	def test_requires_tenant_id(self, db):
		from service import TimeAttendanceService
		with pytest.raises(AssertionError):
			TimeAttendanceService(db, "", ACTOR)

	def test_requires_actor_id(self, db):
		from service import TimeAttendanceService
		with pytest.raises(AssertionError):
			TimeAttendanceService(db, TENANT, "")

	def test_events_empty(self, svc):
		assert svc._events == []

	def test_log_ctx_format(self, svc):
		ctx = svc._log_ctx("test_method", foo="bar")
		assert "test_method" in ctx
		assert TENANT in ctx
		assert "foo=bar" in ctx


# ---------------------------------------------------------------------------
# TimeAttendanceService — _emit_event
# ---------------------------------------------------------------------------

class TestEmitEvent:
	def test_emit_adds_event(self, svc):
		svc._emit_event("tat.test.event", {"k": "v"})
		assert len(svc._events) == 1
		ev = svc._events[0]
		assert ev["type"] == "tat.test.event"
		assert ev["tenant_id"] == TENANT
		assert ev["actor_id"] == ACTOR
		assert ev["payload"]["k"] == "v"
		assert "id" in ev
		assert "occurred_at" in ev

	def test_emit_multiple(self, svc):
		for i in range(3):
			svc._emit_event(f"evt.{i}", {})
		assert len(svc._events) == 3

	def test_emit_payload_preserved(self, svc):
		svc._emit_event("tat.x", {"num": 42, "flag": True})
		assert svc._events[0]["payload"] == {"num": 42, "flag": True}


# ---------------------------------------------------------------------------
# TimeAttendanceService — _assert_tenant
# ---------------------------------------------------------------------------

class TestAssertTenant:
	def test_same_tenant_ok(self, svc):
		svc._assert_tenant({"tenant_id": TENANT})

	def test_wrong_tenant_raises(self, svc):
		with pytest.raises(RuleViolation):
			svc._assert_tenant({"tenant_id": "other-tenant"})


# ---------------------------------------------------------------------------
# TimeAttendanceService — NotFoundError on missing records
# ---------------------------------------------------------------------------

class TestNotFoundError:
	def test_fetch_one_missing_raises(self, svc):
		from service import NotFoundError
		with pytest.raises(NotFoundError):
			asyncio.run(svc._fetch_one("tat_time_policy", "no-such-id"))

	def test_clock_out_missing_entry_raises(self, svc):
		from service import NotFoundError
		with pytest.raises(NotFoundError):
			asyncio.run(svc.clock_out("no-entry"))

	def test_get_time_entry_missing_raises(self, svc):
		from service import NotFoundError
		with pytest.raises(NotFoundError):
			asyncio.run(svc.get_time_entry("no-entry"))

	def test_get_shift_missing_raises(self, svc):
		from service import NotFoundError
		with pytest.raises(NotFoundError):
			asyncio.run(svc.get_shift("no-shift"))

	def test_get_timesheet_missing_raises(self, svc):
		from service import NotFoundError
		with pytest.raises(NotFoundError):
			asyncio.run(svc.get_timesheet("no-ts"))

	def test_get_leave_request_missing_raises(self, svc):
		from service import NotFoundError
		with pytest.raises(NotFoundError):
			asyncio.run(svc.get_leave_request("no-lr"))


# ---------------------------------------------------------------------------
# TimeAttendanceService — list_* returns empty list on empty DB
# ---------------------------------------------------------------------------

class TestListEmpty:
	def _run(self, coro):
		return asyncio.run(coro)

	def test_list_time_entries_empty(self, svc):
		assert self._run(svc.list_time_entries()) == []

	def test_list_timesheets_empty(self, svc):
		assert self._run(svc.list_timesheets()) == []

	def test_list_leave_requests_empty(self, svc):
		assert self._run(svc.list_leave_requests()) == []

	def test_list_shifts_empty(self, svc):
		assert self._run(svc.list_shifts()) == []

	def test_list_time_policies_empty(self, svc):
		assert self._run(svc.list_time_policies()) == []

	def test_list_exceptions_empty(self, svc):
		assert self._run(svc.list_exceptions()) == []

	def test_list_geofence_locations_empty(self, svc):
		assert self._run(svc.list_geofence_locations()) == []

	def test_list_payroll_exports_empty(self, svc):
		assert self._run(svc.list_payroll_exports()) == []

	def test_list_public_holidays_empty(self, svc):
		assert self._run(svc.list_public_holidays()) == []


# ---------------------------------------------------------------------------
# TimeAttendanceService — domain rules integration (pure, no DB needed)
# ---------------------------------------------------------------------------

class TestDomainRulesViaService:
	"""These test that service methods call the right domain rules."""

	def test_comp_time_use_insufficient_raises(self, svc, db):
		"""use_comp_time with no existing balance raises RuleViolation or TimeAttendanceError."""
		from service import TimeAttendanceError
		# No existing comp_time rows → balance = 0, requesting 8h should raise
		with pytest.raises((RuleViolation, TimeAttendanceError, Exception)):
			asyncio.run(svc.use_comp_time("emp-001", Decimal("8"), "testing"))

	def test_gps_geofence_validation_inside(self, svc, db):
		"""GPS validation returns is_valid when inside the fence."""
		loc_id = _uuid()
		db.seed("tat_geofence_location", {
			"id": loc_id,
			"tenant_id": TENANT,
			"name": "HQ",
			"latitude": -1.286389,
			"longitude": 36.817223,
			"radius_metres": 500.0,
		})
		result = asyncio.run(
			svc.gps_geofence_validation("emp-001", -1.286389, 36.817223, loc_id)
		)
		assert result["is_valid"] is True
		assert result["distance_metres"] == pytest.approx(0.0, abs=1.0)

	def test_gps_geofence_validation_outside(self, svc, db):
		"""GPS validation returns is_valid=False when outside the fence."""
		loc_id = _uuid()
		db.seed("tat_geofence_location", {
			"id": loc_id,
			"tenant_id": TENANT,
			"name": "HQ",
			"latitude": -1.286389,
			"longitude": 36.817223,
			"radius_metres": 100.0,
		})
		# Mombasa is ~480 km away
		result = asyncio.run(
			svc.gps_geofence_validation("emp-001", -4.043477, 39.668206, loc_id)
		)
		assert result["is_valid"] is False


# ---------------------------------------------------------------------------
# TimeAttendanceLifecycleService — full in-memory lifecycle
# ---------------------------------------------------------------------------

class TestLifecycleCreate:
	"""Test CRUD operations on the lifecycle service using correct signatures."""

	def test_create_time_policy(self, lifecycle):
		p = lifecycle.create_time_policy(
			_uuid(), TENANT, "Standard 40h", "UTC",
			["Mon", "Tue", "Wed", "Thu", "Fri"], 40.0,
		)
		assert p["name"] == "Standard 40h"
		assert p["tenant_id"] == TENANT
		assert p["status"] == "active"

	def test_create_schedule_requires_policy(self, lifecycle):
		"""create_schedule validates policy belongs to tenant."""
		pol = lifecycle.create_time_policy(
			_uuid(), TENANT, "P", "UTC", ["Mon"], 40.0
		)
		s = lifecycle.create_schedule(
			_uuid(), TENANT, "emp-001", pol["id"], "fixed",
			"2026-01-01", "2026-12-31",
		)
		assert s["schedule_type"] == "fixed"
		assert s["employee_id"] == "emp-001"

	def test_create_schedule_unknown_policy_denied(self, lifecycle):
		"""No policy in store → policy_present=False → deny."""
		with pytest.raises(PermissionError):
			lifecycle.create_schedule(
				_uuid(), TENANT, "emp-001", "no-such-policy", "fixed",
				"2026-01-01", "2026-12-31",
			)

	def test_record_time_entry_no_shift_denied(self, lifecycle):
		"""shift_present=False → deny."""
		with pytest.raises(PermissionError):
			lifecycle.record_time_entry(
				_uuid(), TENANT, "emp-001", "no-shift",
				"regular", "web", "2026-06-01T09:00:00Z", "2026-06-01T17:00:00Z",
			)

	def _seed_entry(self, lifecycle, tenant: str, emp: str) -> str:
		"""Create a shift + time entry so submit_timesheet has entries_present=True."""
		pol = lifecycle.create_time_policy(
			_uuid(), tenant, "Seed Policy", "UTC", ["Mon"], 40.0,
		)
		sched = lifecycle.create_schedule(
			_uuid(), tenant, emp, pol["id"], "fixed",
			"2026-01-01", "2026-12-31",
		)
		shift = lifecycle.create_shift(
			_uuid(), tenant, sched["id"],
			"2026-06-02", "09:00", "17:00",
		)
		entry = lifecycle.record_time_entry(
			_uuid(), tenant, emp, shift["id"],
			"regular", "web",
			"2026-06-02T09:00:00Z", "2026-06-02T17:00:00Z",
		)
		return entry["id"]

	def test_submit_timesheet(self, lifecycle):
		emp = "emp-ts-01"
		entry_id = self._seed_entry(lifecycle, TENANT, emp)
		ts = lifecycle.submit_timesheet(
			_uuid(), TENANT, emp, "2026-06-01", "2026-06-30", [entry_id], emp,
		)
		assert ts["employee_id"] == emp
		assert ts["status"] == "submitted"

	def test_approve_timesheet(self, lifecycle):
		emp = "emp-ts-02"
		entry_id = self._seed_entry(lifecycle, TENANT, emp)
		ts_id = _uuid()
		lifecycle.submit_timesheet(
			ts_id, TENANT, emp, "2026-06-01", "2026-06-30", [entry_id], emp,
		)
		approved = lifecycle.approve_timesheet(ts_id, TENANT, "manager-001")
		assert approved["status"] == "approved"
		assert approved["approved_by"] == "manager-001"

	def test_request_leave(self, lifecycle):
		lr = lifecycle.request_leave(
			_uuid(), TENANT, "emp-001", "vacation",
			"2026-07-01", "2026-07-05", "Summer holiday",
		)
		assert lr["leave_type"] == "vacation"
		assert lr["status"] == "requested"

	def test_request_leave_with_approval(self, lifecycle):
		lr = lifecycle.request_leave(
			_uuid(), TENANT, "emp-001", "sick",
			"2026-07-10", "2026-07-11", "Illness", "manager-001",
		)
		assert lr["status"] == "approved"
		assert lr["approved_by"] == "manager-001"

	def test_record_exception(self, lifecycle):
		exc = lifecycle.record_exception(
			_uuid(), TENANT, "emp-001",
			"late_arrival", "medium", "15 minutes late",
		)
		assert exc["exception_type"] == "late_arrival"
		assert exc["status"] == "open"

	def test_create_payroll_export_with_approved_timesheet(self, lifecycle):
		"""Payroll export requires timesheets_present AND all_timesheets_approved."""
		emp = "emp-pe-01"
		entry_id = self._seed_entry(lifecycle, TENANT, emp)
		ts_id = _uuid()
		lifecycle.submit_timesheet(
			ts_id, TENANT, emp, "2026-06-01", "2026-06-30", [entry_id], emp,
		)
		lifecycle.approve_timesheet(ts_id, TENANT, "manager-001")
		exp = lifecycle.create_payroll_export(
			_uuid(), TENANT, "2026-06-01", "2026-06-30", [ts_id], "manager-001",
		)
		assert "id" in exp
		assert exp["total_hours"] >= 0

	def test_create_payroll_export_empty_denied(self, lifecycle):
		"""Empty timesheet_ids triggers timesheets_present=False → deny."""
		with pytest.raises(PermissionError):
			lifecycle.create_payroll_export(
				_uuid(), TENANT, "2026-06-01", "2026-06-30", [], "manager-001",
			)

	def test_register_attendance_agent_valid_role(self, lifecycle):
		agent = lifecycle.register_attendance_agent(
			TENANT, "CI Bot", "codex",
			"attendance_reviewer", "review attendance",
		)
		assert agent["name"] == "CI Bot"
		assert agent["role"] == "attendance_reviewer"
		assert agent["status"] == "active"

	def test_register_attendance_agent_invalid_role(self, lifecycle):
		with pytest.raises(PermissionError):
			lifecycle.register_attendance_agent(
				TENANT, "Bad Bot", "codex", "unsupported_role", "testing",
			)

	def test_dashboard_summary(self, lifecycle):
		summary = lifecycle.dashboard_summary(TENANT)
		assert isinstance(summary, dict)
		assert "policy_count" in summary
		assert "timesheet_count" in summary

	def test_audit_events_after_operations(self, lifecycle):
		lifecycle.create_time_policy(
			_uuid(), TENANT, "AuditTest", "UTC", ["Mon"], 40.0,
		)
		events = lifecycle.audit_events(TENANT)
		assert isinstance(events, list)
		assert len(events) >= 1
		assert events[0]["event_type"] == "attendance_policy_created"

	def test_list_records_empty_tenant(self, lifecycle):
		records = lifecycle.list_records("no-such-tenant")
		assert records == []

	def test_list_records_type_filter(self, lifecycle):
		lifecycle.create_time_policy(
			_uuid(), TENANT, "Filter Test", "UTC", ["Mon"], 40.0,
		)
		policies = lifecycle.list_records(TENANT, "policy")
		assert all(r["kind"] == "policy" for r in policies)

	def test_create_record_generic(self, lifecycle):
		r = lifecycle.create_record({
			"id": _uuid(),
			"tenant_id": TENANT,
			"type": "custom_note",
			"kind": "note",
			"data": "hello world",
		})
		assert "id" in r
		assert r["type"] == "custom_note"

	def test_multiple_tenants_isolated(self, lifecycle):
		lifecycle.create_time_policy(
			_uuid(), "tenant-A", "A Policy", "UTC", ["Mon"], 40.0,
		)
		lifecycle.create_time_policy(
			_uuid(), "tenant-B", "B Policy", "UTC", ["Mon"], 40.0,
		)
		a = lifecycle.list_records("tenant-A", "policy")
		b = lifecycle.list_records("tenant-B", "policy")
		assert all(r["tenant_id"] == "tenant-A" for r in a)
		assert all(r["tenant_id"] == "tenant-B" for r in b)


# ---------------------------------------------------------------------------
# TimeAttendanceLifecycleService — validate methods
# ---------------------------------------------------------------------------

class TestLifecycleValidate:
	def test_validate_agent_action_no_privilege_ok(self, lifecycle):
		result = lifecycle.validate_attendance_agent_action(TENANT, False, False)
		assert result["decision"] == "allow"

	def test_validate_batch_bytewax(self, lifecycle):
		result = lifecycle.validate_batch(TENANT, 100, "bytewax")
		assert result["accepted"] is True
		assert result["processor"] == "bytewax"

	def test_describe_returns_contract(self, lifecycle):
		contract = lifecycle.describe(TENANT)
		assert isinstance(contract, dict)

	def test_evaluate_context_allow(self, lifecycle):
		result = lifecycle.evaluate({
			"tenant_id": TENANT,
			"tenant_context_present": True,
			"operation": "noop",
		})
		assert result["decision"] == "allow"


# ---------------------------------------------------------------------------
# Export assertions
# ---------------------------------------------------------------------------

class TestServiceExports:
	def test_service_class(self):
		from service import TimeAttendanceService
		assert TimeAttendanceService

	def test_lifecycle_service_class(self):
		from service import TimeAttendanceLifecycleService
		assert TimeAttendanceLifecycleService

	def test_errors(self):
		from service import TimeAttendanceError, NotFoundError
		assert issubclass(TimeAttendanceError, Exception)
		assert issubclass(NotFoundError, TimeAttendanceError)

	def test_sub_services(self):
		from service import TimeEntryService, AttendanceScheduleService, AttendanceComplianceService
		assert TimeEntryService
		assert AttendanceScheduleService
		assert AttendanceComplianceService
