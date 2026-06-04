"""
CI tests for api.py Flask Blueprint (TAT REST API).

Covers:
1. /health endpoint (always live)
2. Legacy dict-API helper functions (use in-memory lifecycle service, no DB)
3. Blueprint URL structure (routes exist, return structured JSON)
4. Blueprint error handling (stub DB → 500 with structured ok=False body)

Copyright © 2025 Datacraft. Author: Nyimbi Odero
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from flask import Flask
from uuid6 import uuid7

TENANT = "tenant-ci"
ACTOR = "actor-ci"


def _uuid() -> str:
	return str(uuid7())


# ---------------------------------------------------------------------------
# App fixture — register the blueprint once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app() -> Flask:
	flask_app = Flask(__name__)
	flask_app.config["TESTING"] = True
	from api import bp
	flask_app.register_blueprint(bp)
	return flask_app


@pytest.fixture(scope="module")
def client(app: Flask):
	return app.test_client()


def _hdr() -> dict[str, str]:
	return {
		"X-Tenant-Id": TENANT,
		"X-Actor-Id": ACTOR,
		"Content-Type": "application/json",
	}


# ---------------------------------------------------------------------------
# Health (no DB, always succeeds)
# ---------------------------------------------------------------------------

class TestHealth:
	def test_health_200(self, client):
		r = client.get("/hcm/time-attendance/api/v1/health")
		assert r.status_code == 200

	def test_health_body(self, client):
		data = client.get("/hcm/time-attendance/api/v1/health").get_json()
		assert data["ok"] is True
		assert data["data"]["status"] == "ok"
		assert data["data"]["capability"] == "tat_time_attendance"


# ---------------------------------------------------------------------------
# Legacy dict-API helpers (in-memory lifecycle service, no DB)
# ---------------------------------------------------------------------------

class TestLegacyHelpers:
	"""
	api.py exports thin wrappers around TimeAttendanceLifecycleService.
	These need correct positional payloads and seeded state where required.
	"""

	def _policy_payload(self) -> dict[str, Any]:
		return {
			"id": _uuid(),
			"tenant_id": TENANT,
			"name": "Standard 40h",
			"timezone": "UTC",
			"workweek": ["Mon", "Tue", "Wed", "Thu", "Fri"],
			"overtime_threshold_hours": 40.0,
		}

	def test_create_time_policy(self):
		from api import create_time_policy
		r = create_time_policy(self._policy_payload())
		assert r["name"] == "Standard 40h"
		assert "id" in r

	def test_create_time_policy_returns_active(self):
		from api import create_time_policy
		r = create_time_policy(self._policy_payload())
		assert r["status"] == "active"

	def _seed_entry_for_timesheet(self) -> tuple[str, str]:
		"""Seed a shift + time entry into the shared legacy service, return (entry_id, emp_id)."""
		from api import _LEGACY
		from uuid6 import uuid7
		emp = f"emp-ts-api-{str(uuid7())[:8]}"
		pol = _LEGACY.create_time_policy(
			str(uuid7()), TENANT, "API Test Policy", "UTC", ["Mon"], 40.0,
		)
		sched = _LEGACY.create_schedule(
			str(uuid7()), TENANT, emp, pol["id"], "fixed",
			"2026-01-01", "2026-12-31",
		)
		shift = _LEGACY.create_shift(
			str(uuid7()), TENANT, sched["id"],
			"2026-06-02", "09:00", "17:00",
		)
		entry = _LEGACY.record_time_entry(
			str(uuid7()), TENANT, emp, shift["id"],
			"regular", "web",
			"2026-06-02T09:00:00Z", "2026-06-02T17:00:00Z",
		)
		return entry["id"], emp

	def test_submit_timesheet(self):
		from api import submit_timesheet
		entry_id, emp = self._seed_entry_for_timesheet()
		r = submit_timesheet({
			"id": _uuid(),
			"tenant_id": TENANT,
			"employee_id": emp,
			"period_start": "2026-06-01",
			"period_end": "2026-06-30",
			"entry_ids": [entry_id],
			"submitted_by": emp,
		})
		assert "id" in r
		assert r["employee_id"] == emp

	def test_approve_timesheet(self):
		from api import submit_timesheet, approve_timesheet
		entry_id, emp = self._seed_entry_for_timesheet()
		ts_id = _uuid()
		submit_timesheet({
			"id": ts_id,
			"tenant_id": TENANT,
			"employee_id": emp,
			"period_start": "2026-06-01",
			"period_end": "2026-06-30",
			"entry_ids": [entry_id],
			"submitted_by": emp,
		})
		r = approve_timesheet({
			"id": ts_id,
			"tenant_id": TENANT,
			"approved_by": "manager-001",
		})
		assert r["status"] == "approved"

	def test_request_leave(self):
		from api import request_leave
		r = request_leave({
			"id": _uuid(),
			"tenant_id": TENANT,
			"employee_id": "emp-001",
			"leave_type": "vacation",
			"start_date": "2026-07-01",
			"end_date": "2026-07-05",
			"reason": "Annual holiday",
		})
		assert r["leave_type"] == "vacation"
		assert "id" in r

	def test_record_exception(self):
		from api import record_exception
		r = record_exception({
			"id": _uuid(),
			"tenant_id": TENANT,
			"employee_id": "emp-001",
			"exception_type": "late_arrival",
			"severity": "medium",
			"description": "15 minutes late",
		})
		assert r["exception_type"] == "late_arrival"
		assert r["status"] == "open"

	def test_create_payroll_export(self):
		"""Payroll export with empty timesheets is denied by capability rules."""
		from api import create_payroll_export
		with pytest.raises(PermissionError):
			create_payroll_export({
				"id": _uuid(),
				"tenant_id": TENANT,
				"period_start": "2026-06-01",
				"period_end": "2026-06-30",
				"timesheet_ids": [],
				"approved_by": "manager-001",
				"event_stream": "bytewax",
			})

	def test_create_payroll_export_with_approved_timesheet(self):
		"""Payroll export succeeds with an approved timesheet."""
		from api import create_payroll_export, approve_timesheet, submit_timesheet
		entry_id, emp = self._seed_entry_for_timesheet()
		ts_id = _uuid()
		submit_timesheet({
			"id": ts_id, "tenant_id": TENANT, "employee_id": emp,
			"period_start": "2026-06-01", "period_end": "2026-06-30",
			"entry_ids": [entry_id], "submitted_by": emp,
		})
		approve_timesheet({"id": ts_id, "tenant_id": TENANT, "approved_by": "manager-001"})
		r = create_payroll_export({
			"id": _uuid(), "tenant_id": TENANT,
			"period_start": "2026-06-01", "period_end": "2026-06-30",
			"timesheet_ids": [ts_id], "approved_by": "manager-001",
			"event_stream": "bytewax",
		})
		assert "id" in r

	def test_dashboard_summary(self):
		from api import dashboard_summary
		r = dashboard_summary(TENANT)
		assert isinstance(r, dict)
		assert "policy_count" in r

	def test_audit_events(self):
		from api import audit_events
		r = audit_events(TENANT)
		assert isinstance(r, list)

	def test_register_attendance_agent(self):
		from api import register_attendance_agent
		r = register_attendance_agent({
			"tenant_id": TENANT,
			"name": "CI Agent",
			"runtime": "codex",
			"role": "attendance_reviewer",
			"purpose": "CI testing",
		})
		assert "id" in r
		assert r["role"] == "attendance_reviewer"

	def test_create_record_generic(self):
		from api import create_record
		r = create_record({
			"id": _uuid(),
			"tenant_id": TENANT,
			"type": "custom_note",
			"kind": "note",
		})
		assert "id" in r

	def test_service_returns_lifecycle_instance(self):
		from api import service
		from lifecycle import TimeAttendanceLifecycleService
		assert isinstance(service(), TimeAttendanceLifecycleService)


# ---------------------------------------------------------------------------
# Blueprint — URL structure (no DB; endpoints that call service return 500)
# The test only asserts route is registered (not 404).
# ---------------------------------------------------------------------------

REGISTERED_ROUTES = [
	("GET",  "/hcm/time-attendance/api/v1/health"),
	("GET",  "/hcm/time-attendance/api/v1/policies"),
	("POST", "/hcm/time-attendance/api/v1/policies"),
	("GET",  "/hcm/time-attendance/api/v1/schedules"),
	("POST", "/hcm/time-attendance/api/v1/schedules"),
	("GET",  "/hcm/time-attendance/api/v1/shifts"),
	("POST", "/hcm/time-attendance/api/v1/shifts"),
	("GET",  "/hcm/time-attendance/api/v1/time-entries"),
	("POST", "/hcm/time-attendance/api/v1/time-entries/clock-in"),
	("GET",  "/hcm/time-attendance/api/v1/timesheets"),
	("POST", "/hcm/time-attendance/api/v1/timesheets/process"),
	("POST", "/hcm/time-attendance/api/v1/timesheets/bulk-import"),
	("GET",  "/hcm/time-attendance/api/v1/leave"),
	("POST", "/hcm/time-attendance/api/v1/leave"),
	("GET",  "/hcm/time-attendance/api/v1/geofences"),
	("POST", "/hcm/time-attendance/api/v1/geofences"),
	("POST", "/hcm/time-attendance/api/v1/geofences/validate"),
	("GET",  "/hcm/time-attendance/api/v1/exceptions"),
	("POST", "/hcm/time-attendance/api/v1/exceptions"),
	("GET",  "/hcm/time-attendance/api/v1/payroll-exports"),
	("POST", "/hcm/time-attendance/api/v1/payroll-exports"),
	("GET",  "/hcm/time-attendance/api/v1/public-holidays"),
	("POST", "/hcm/time-attendance/api/v1/public-holidays"),
	("GET",  "/hcm/time-attendance/api/v1/overtime"),
	("POST", "/hcm/time-attendance/api/v1/overtime/requests"),
	("POST", "/hcm/time-attendance/api/v1/shifts/swap"),
	("POST", "/hcm/time-attendance/api/v1/comp-time/earn"),
	("POST", "/hcm/time-attendance/api/v1/comp-time/use"),
	("POST", "/hcm/time-attendance/api/v1/rosters/generate"),
	("GET",  "/hcm/time-attendance/api/v1/dashboard"),
	("GET",  "/hcm/time-attendance/api/v1/reports/daily_summary"),
]


class TestUrlStructure:
	@pytest.mark.parametrize("method,url", REGISTERED_ROUTES)
	def test_route_registered(self, client, method: str, url: str):
		r = client.open(url, method=method, headers=_hdr(),
						data=json.dumps({}))
		assert r.status_code != 404, f"{method} {url} returned 404 — route missing"

	@pytest.mark.parametrize("method,url", REGISTERED_ROUTES)
	def test_response_is_json(self, client, method: str, url: str):
		r = client.open(url, method=method, headers=_hdr(),
						data=json.dumps({}))
		data = r.get_json()
		assert data is not None, f"{method} {url} returned non-JSON"

	@pytest.mark.parametrize("method,url", REGISTERED_ROUTES)
	def test_response_has_ok_key(self, client, method: str, url: str):
		r = client.open(url, method=method, headers=_hdr(),
						data=json.dumps({}))
		data = r.get_json()
		assert "ok" in data, f"{method} {url} response missing 'ok' key: {data}"


# ---------------------------------------------------------------------------
# Blueprint — error handling
# ---------------------------------------------------------------------------

class TestBlueprintErrorHandling:
	def test_missing_entry_returns_structured_error(self, client):
		r = client.get(
			"/hcm/time-attendance/api/v1/time-entries/no-such-id",
			headers=_hdr(),
		)
		assert r.status_code in (404, 500)
		data = r.get_json()
		assert data["ok"] is False

	def test_missing_policy_returns_structured_error(self, client):
		r = client.get(
			"/hcm/time-attendance/api/v1/policies/no-such-id",
			headers=_hdr(),
		)
		assert r.status_code in (404, 500)
		data = r.get_json()
		assert data["ok"] is False

	def test_clock_in_no_db_structured_error(self, client):
		payload = json.dumps({
			"employee_id": "emp-001",
			"shift_id": "shift-001",
			"method": "web",
		})
		r = client.post(
			"/hcm/time-attendance/api/v1/time-entries/clock-in",
			data=payload,
			headers=_hdr(),
		)
		assert r.status_code in (400, 422, 500)
		assert r.get_json()["ok"] is False

	def test_health_always_200(self, client):
		"""Health endpoint should always return 200 regardless of DB state."""
		r = client.get("/hcm/time-attendance/api/v1/health")
		assert r.status_code == 200
		assert r.get_json()["ok"] is True
