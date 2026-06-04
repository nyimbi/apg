"""
CI tests for models.py

Tests Pydantic v2 model construction, validation, computed fields, and
enum membership. No DB, no async, no mocks.

Copyright © 2025 Datacraft. Author: Nyimbi Odero
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from models import (
	# Enums
	AIAgentType,
	ApprovalStatus,
	AttendanceStatus,
	BiometricType,
	DeviceType,
	FraudType,
	LeaveType,
	ProductivityMetric,
	RemoteWorkStatus,
	ScheduleStatus,
	TimeEntryStatus,
	TimeEntryType,
	WorkMode,
	WorkforceType,
	# Models
	TABaseModel,
	TAEmployee,
	TATimeEntry,
	TASchedule,
	TALeaveRequest,
	TAFraudDetection,
	TABiometricAuthentication,
	TARemoteWorker,
	TAAIAgent,
	TAHybridCollaboration,
	# Validators
	_validate_confidence_score,
	_validate_geolocation,
)

UTC = timezone.utc
_TENANT = "tenant-test"
_ACTOR = "actor-001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt(h: int, d: int = 1) -> datetime:
	return datetime(2026, 6, d, h, 0, tzinfo=UTC)


def _employee(**kw) -> TAEmployee:
	defaults = dict(
		tenant_id=_TENANT,
		employee_id="emp-001",
		employee_number="E001",
		department_id="dept-001",
		created_by=_ACTOR,
	)
	defaults.update(kw)
	return TAEmployee(**defaults)


# ---------------------------------------------------------------------------
# Validator functions
# ---------------------------------------------------------------------------

class TestValidateConfidenceScore:
	def test_zero_ok(self):
		assert _validate_confidence_score(0.0) == 0.0

	def test_one_ok(self):
		assert _validate_confidence_score(1.0) == 1.0

	def test_mid_ok(self):
		assert _validate_confidence_score(0.75) == 0.75

	def test_below_zero_raises(self):
		with pytest.raises(ValueError):
			_validate_confidence_score(-0.01)

	def test_above_one_raises(self):
		with pytest.raises(ValueError):
			_validate_confidence_score(1.01)


class TestValidateGeolocation:
	def test_valid_nairobi(self):
		result = _validate_geolocation({"latitude": -1.286, "longitude": 36.820})
		assert result["latitude"] == -1.286

	def test_missing_key_raises(self):
		with pytest.raises(ValueError):
			_validate_geolocation({"latitude": 0.0})

	def test_lat_too_high_raises(self):
		with pytest.raises(ValueError):
			_validate_geolocation({"latitude": 91.0, "longitude": 0.0})

	def test_lat_too_low_raises(self):
		with pytest.raises(ValueError):
			_validate_geolocation({"latitude": -91.0, "longitude": 0.0})

	def test_lng_too_high_raises(self):
		with pytest.raises(ValueError):
			_validate_geolocation({"latitude": 0.0, "longitude": 181.0})


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
	def test_time_entry_status_values(self):
		assert TimeEntryStatus.DRAFT.value == "draft"
		assert TimeEntryStatus.APPROVED.value == "approved"
		assert TimeEntryStatus.LOCKED.value == "locked"

	def test_time_entry_type_values(self):
		assert TimeEntryType.REGULAR.value == "regular"
		assert TimeEntryType.OVERTIME.value == "overtime"
		assert TimeEntryType.SICK.value == "sick"

	def test_attendance_status_values(self):
		assert AttendanceStatus.PRESENT.value == "present"
		assert AttendanceStatus.LATE.value == "late"
		assert AttendanceStatus.REMOTE.value == "remote"

	def test_leave_type_values(self):
		assert LeaveType.VACATION.value == "vacation"
		assert LeaveType.SICK.value == "sick"
		assert LeaveType.SABBATICAL.value == "sabbatical"
		assert LeaveType.MATERNITY.value == "maternity"
		assert LeaveType.PATERNITY.value == "paternity"

	def test_approval_status_values(self):
		assert ApprovalStatus.PENDING.value == "pending"
		assert ApprovalStatus.APPROVED.value == "approved"
		assert ApprovalStatus.REJECTED.value == "rejected"

	def test_schedule_status_values(self):
		assert ScheduleStatus.DRAFT.value == "draft"
		assert ScheduleStatus.PUBLISHED.value == "published"

	def test_biometric_type_values(self):
		assert BiometricType.FINGERPRINT.value == "fingerprint"
		assert BiometricType.FACIAL_RECOGNITION.value == "facial_recognition"

	def test_device_type_values(self):
		assert DeviceType.MOBILE_APP.value == "mobile_app"
		assert DeviceType.BIOMETRIC_TERMINAL.value == "biometric_terminal"
		assert DeviceType.WEB_BROWSER.value == "web_browser"

	def test_fraud_type_values(self):
		assert FraudType.BUDDY_PUNCHING.value == "buddy_punching"
		assert FraudType.LOCATION_SPOOFING.value == "location_spoofing"

	def test_workforce_type_values(self):
		assert WorkforceType.HUMAN_EMPLOYEE.value == "human_employee"
		assert WorkforceType.AI_AGENT.value == "ai_agent"

	def test_work_mode_values(self):
		assert WorkMode.OFFICE_BASED.value == "office_based"
		assert WorkMode.REMOTE_ONLY.value == "remote_only"
		assert WorkMode.HYBRID.value == "hybrid"

	def test_ai_agent_type_values(self):
		assert AIAgentType.CONVERSATIONAL_AI.value == "conversational_ai"
		assert AIAgentType.AUTOMATION_BOT.value == "automation_bot"
		assert AIAgentType.ANALYSIS_AGENT.value == "analysis_agent"

	def test_productivity_metric_values(self):
		assert ProductivityMetric.TIME_BASED.value == "time_based"
		assert ProductivityMetric.TASK_COMPLETION.value == "task_completion"
		assert ProductivityMetric.OUTPUT_QUALITY.value == "output_quality"

	def test_remote_work_status_values(self):
		assert RemoteWorkStatus.ACTIVE_WORKING.value == "active_working"
		assert RemoteWorkStatus.IN_MEETING.value == "in_meeting"
		assert RemoteWorkStatus.OFFLINE.value == "offline"


# ---------------------------------------------------------------------------
# TABaseModel
# ---------------------------------------------------------------------------

class TestTABaseModel:
	def test_id_generated(self):
		# TABaseModel is abstract; instantiate via a concrete subclass
		emp = _employee()
		assert emp.id is not None
		assert len(emp.id) > 8

	def test_tenant_id_required(self):
		with pytest.raises(ValidationError):
			TAEmployee(employee_id="e", employee_number="E1", department_id="d", created_by="a")

	def test_extra_fields_forbidden(self):
		with pytest.raises(ValidationError):
			_employee(unknown_field="x")


# ---------------------------------------------------------------------------
# TAEmployee
# ---------------------------------------------------------------------------

class TestTAEmployee:
	def test_minimal_construction(self):
		emp = _employee()
		assert emp.tenant_id == _TENANT
		assert emp.employee_id == "emp-001"
		assert emp.biometric_enabled is False
		assert emp.fraud_risk_score == 0.0

	def test_has_active_biometrics_false_by_default(self):
		emp = _employee()
		assert emp.has_active_biometrics is False

	def test_has_active_biometrics_true(self):
		template = {
			"type": "fingerprint",
			"template_data": "A" * 64,
			"created_at": "2026-01-01T00:00:00",
			"quality_score": 0.95,
		}
		emp = _employee(
			biometric_enabled=True,
			biometric_consent=True,
			biometric_templates=[template],
		)
		assert emp.has_active_biometrics is True

	def test_fraud_risk_score_bounds(self):
		with pytest.raises(ValidationError):
			_employee(fraud_risk_score=1.5)

	def test_is_active_default_true(self):
		assert _employee().is_active is True

	def test_timezone_default_utc(self):
		assert _employee().timezone == "UTC"


# ---------------------------------------------------------------------------
# TATimeEntry
# ---------------------------------------------------------------------------

class TestTATimeEntry:
	def _entry(self, **kw) -> TATimeEntry:
		defaults = dict(
			tenant_id=_TENANT,
			employee_id="emp-001",
			entry_date=date(2026, 6, 1),
			clock_in=_dt(9),
			created_by=_ACTOR,
		)
		defaults.update(kw)
		return TATimeEntry(**defaults)

	def test_minimal_construction(self):
		entry = self._entry()
		assert entry.employee_id == "emp-001"
		assert entry.status == TimeEntryStatus.DRAFT
		assert entry.entry_type == TimeEntryType.REGULAR

	def test_duration_hours_computed(self):
		entry = self._entry(clock_out=_dt(17))
		assert entry.duration_hours == pytest.approx(8.0)

	def test_duration_hours_none_without_clock_out(self):
		entry = self._entry()
		assert entry.duration_hours is None

	def test_clock_out_before_in_raises(self):
		with pytest.raises(ValidationError):
			self._entry(clock_in=_dt(17), clock_out=_dt(9))

	def test_clock_out_without_clock_in_raises(self):
		with pytest.raises(ValidationError):
			TATimeEntry(
				tenant_id=_TENANT,
				employee_id="emp-001",
				entry_date=date(2026, 6, 1),
				clock_out=_dt(17),
				created_by=_ACTOR,
			)

	def test_is_overtime_eligible_true(self):
		entry = self._entry(total_hours=Decimal("9"))
		assert entry.is_overtime_eligible is True

	def test_is_overtime_eligible_false_short_day(self):
		entry = self._entry(total_hours=Decimal("7"))
		assert entry.is_overtime_eligible is False

	def test_anomaly_score_bounds(self):
		with pytest.raises(ValidationError):
			self._entry(anomaly_score=1.5)

	def test_geolocation_validation(self):
		entry = self._entry(clock_in_location={"latitude": -1.286, "longitude": 36.820})
		assert entry.clock_in_location["latitude"] == pytest.approx(-1.286)

	def test_geolocation_invalid_raises(self):
		with pytest.raises(ValidationError):
			self._entry(clock_in_location={"latitude": 200.0, "longitude": 0.0})


# ---------------------------------------------------------------------------
# TASchedule
# ---------------------------------------------------------------------------

class TestTASchedule:
	def _pattern(self) -> dict:
		return {"days_of_week": [0, 1, 2, 3, 4], "start_time": "09:00", "end_time": "17:00"}

	def _schedule(self, **kw) -> TASchedule:
		defaults = dict(
			tenant_id=_TENANT,
			schedule_name="Standard",
			schedule_type="fixed",
			effective_date=date(2026, 1, 1),
			schedule_patterns=[self._pattern()],
			created_by=_ACTOR,
		)
		defaults.update(kw)
		return TASchedule(**defaults)

	def test_minimal_construction(self):
		s = self._schedule()
		assert s.schedule_name == "Standard"
		assert s.status == ScheduleStatus.DRAFT

	def test_total_weekly_hours(self):
		# 8h/day × 5 days = 40h
		s = self._schedule()
		assert s.total_weekly_hours == pytest.approx(40.0)

	def test_is_active_draft_schedule(self):
		# Draft status → not active
		s = self._schedule()
		assert s.is_active is False

	def test_invalid_pattern_raises(self):
		bad_pattern = {"days_of_week": [0], "start_time": "09:00"}  # missing end_time
		with pytest.raises(ValidationError):
			self._schedule(schedule_patterns=[bad_pattern])

	def test_invalid_days_raises(self):
		bad_pattern = {"days_of_week": [7], "start_time": "09:00", "end_time": "17:00"}
		with pytest.raises(ValidationError):
			self._schedule(schedule_patterns=[bad_pattern])


# ---------------------------------------------------------------------------
# TALeaveRequest
# ---------------------------------------------------------------------------

class TestTALeaveRequest:
	def _leave(self, **kw) -> TALeaveRequest:
		defaults = dict(
			tenant_id=_TENANT,
			employee_id="emp-001",
			leave_type=LeaveType.VACATION,
			start_date=date(2026, 7, 1),
			end_date=date(2026, 7, 5),
			total_days=Decimal("5"),
			total_hours=Decimal("40"),
			created_by=_ACTOR,
		)
		defaults.update(kw)
		return TALeaveRequest(**defaults)

	def test_minimal_construction(self):
		lr = self._leave()
		assert lr.leave_type == LeaveType.VACATION
		assert lr.status == ApprovalStatus.PENDING

	def test_duration_days_computed(self):
		lr = self._leave()
		assert lr.duration_days == 5  # Jul 1–5 inclusive

	def test_is_extended_leave_false(self):
		assert self._leave().is_extended_leave is False

	def test_is_extended_leave_true(self):
		lr = self._leave(
			end_date=date(2026, 7, 14),
			total_days=Decimal("10"),
			total_hours=Decimal("80"),
		)
		assert lr.is_extended_leave is True

	def test_end_before_start_raises(self):
		with pytest.raises(ValidationError):
			self._leave(start_date=date(2026, 7, 5), end_date=date(2026, 7, 1))

	def test_zero_days_raises(self):
		with pytest.raises(ValidationError):
			self._leave(total_days=Decimal("0"))


# ---------------------------------------------------------------------------
# TAFraudDetection
# ---------------------------------------------------------------------------

class TestTAFraudDetection:
	def _fraud(self, **kw) -> TAFraudDetection:
		defaults = dict(
			tenant_id=_TENANT,
			employee_id="emp-001",
			fraud_types=[FraudType.BUDDY_PUNCHING],
			severity_level="HIGH",
			confidence_score=0.9,
			created_by=_ACTOR,
		)
		defaults.update(kw)
		return TAFraudDetection(**defaults)

	def test_minimal_construction(self):
		f = self._fraud()
		assert f.severity_level == "HIGH"
		assert f.confidence_score == 0.9

	def test_requires_immediate_action_high(self):
		f = self._fraud(severity_level="CRITICAL", confidence_score=0.9)
		assert f.requires_immediate_action is True

	def test_requires_immediate_action_low_conf(self):
		f = self._fraud(confidence_score=0.5)
		assert f.requires_immediate_action is False

	def test_risk_level_very_high(self):
		f = self._fraud(severity_level="CRITICAL", confidence_score=0.95)
		assert f.risk_level == "VERY_HIGH"

	def test_risk_level_medium(self):
		f = self._fraud(confidence_score=0.65)
		assert f.risk_level == "MEDIUM"

	def test_invalid_severity_raises(self):
		with pytest.raises(ValidationError):
			self._fraud(severity_level="SEVERE")

	def test_confidence_out_of_range_raises(self):
		with pytest.raises(ValidationError):
			self._fraud(confidence_score=1.5)


# ---------------------------------------------------------------------------
# TABiometricAuthentication
# ---------------------------------------------------------------------------

class TestTABiometricAuthentication:
	def _auth(self, **kw) -> TABiometricAuthentication:
		defaults = dict(
			tenant_id=_TENANT,
			employee_id="emp-001",
			biometric_type=BiometricType.FINGERPRINT,
			device_type=DeviceType.BIOMETRIC_TERMINAL,
			template_quality=0.95,
			authentication_success=True,
			confidence_score=0.92,
			created_by=_ACTOR,
		)
		defaults.update(kw)
		return TABiometricAuthentication(**defaults)

	def test_minimal_construction(self):
		a = self._auth()
		assert a.authentication_success is True
		assert a.template_quality == 0.95

	def test_data_retention_expires_at(self):
		a = self._auth()
		delta = (a.data_retention_expires_at - a.authentication_timestamp).days
		assert delta == a.retention_period_days

	def test_overall_trust_score_biometric_terminal(self):
		a = self._auth(
			device_type=DeviceType.BIOMETRIC_TERMINAL,
			confidence_score=1.0,
			liveness_confidence=1.0,
			template_quality=1.0,
		)
		assert a.overall_trust_score == pytest.approx(1.0)

	def test_invalid_template_quality_raises(self):
		with pytest.raises(ValidationError):
			self._auth(template_quality=1.5)

	def test_retention_bounds(self):
		with pytest.raises(ValidationError):
			self._auth(retention_period_days=0)


# ---------------------------------------------------------------------------
# TARemoteWorker
# ---------------------------------------------------------------------------

class TestTARemoteWorker:
	def _worker(self, **kw) -> TARemoteWorker:
		defaults = dict(
			tenant_id=_TENANT,
			employee_id="emp-001",
			work_mode=WorkMode.REMOTE_ONLY,
			timezone="Africa/Nairobi",
			created_by=_ACTOR,
		)
		defaults.update(kw)
		return TARemoteWorker(**defaults)

	def test_minimal_construction(self):
		w = self._worker()
		assert w.work_mode == WorkMode.REMOTE_ONLY
		assert w.current_activity == RemoteWorkStatus.OFFLINE

	def test_overall_productivity_score_empty(self):
		assert self._worker().overall_productivity_score == 0.0

	def test_overall_productivity_score_with_metrics(self):
		w = self._worker(productivity_metrics=[{"score": 0.8}, {"score": 0.6}])
		assert w.overall_productivity_score == pytest.approx(0.7)

	def test_is_actively_working_false(self):
		assert self._worker().is_actively_working is False

	def test_is_actively_working_true(self):
		w = self._worker(current_activity=RemoteWorkStatus.FOCUSED_WORK)
		assert w.is_actively_working is True

	def test_work_life_balance_bounds(self):
		with pytest.raises(ValidationError):
			self._worker(work_life_balance_score=1.5)


# ---------------------------------------------------------------------------
# TAAIAgent
# ---------------------------------------------------------------------------

class TestTAAIAgent:
	def _agent(self, **kw) -> TAAIAgent:
		defaults = dict(
			tenant_id=_TENANT,
			agent_name="TestBot",
			agent_type=AIAgentType.AUTOMATION_BOT,
			agent_version="1.0",
			capabilities=["automation"],
			configuration={"setting": "value"},
			deployment_environment="production",
			created_by=_ACTOR,
		)
		defaults.update(kw)
		return TAAIAgent(**defaults)

	def test_minimal_construction(self):
		a = self._agent()
		assert a.agent_name == "TestBot"
		assert a.health_status == "healthy"

	def test_cost_efficiency_zero_no_cost(self):
		a = self._agent()
		assert a.cost_efficiency_score == 0.0

	def test_cost_efficiency_positive(self):
		a = self._agent(
			tasks_completed=100,
			total_operational_cost=Decimal("50.00"),
		)
		# human_equivalent = 100 × 25 = 2500; agent = 50 → ratio = 50 (capped at 10)
		assert a.cost_efficiency_score == 10.0

	def test_overall_performance_score(self):
		a = self._agent(
			accuracy_score=0.9,
			efficiency_rating=0.85,
			uptime_percentage=0.99,
			handoff_efficiency=0.8,
		)
		assert 0.0 < a.overall_performance_score <= 1.0

	def test_invalid_health_status_raises(self):
		with pytest.raises(ValidationError):
			self._agent(health_status="broken")

	def test_accuracy_score_bounds(self):
		with pytest.raises(ValidationError):
			self._agent(accuracy_score=1.5)


# ---------------------------------------------------------------------------
# TAHybridCollaboration
# ---------------------------------------------------------------------------

class TestTAHybridCollaboration:
	def _collab(self, **kw) -> TAHybridCollaboration:
		defaults = dict(
			tenant_id=_TENANT,
			session_name="Sprint Planning",
			project_id="proj-001",
			session_type="planning",
			human_participants=["emp-001"],
			ai_participants=["ai-001"],
			session_lead="emp-001",
			start_time=_dt(10),
			planned_duration_minutes=60,
			created_by=_ACTOR,
		)
		defaults.update(kw)
		return TAHybridCollaboration(**defaults)

	def test_minimal_construction(self):
		c = self._collab()
		assert c.session_name == "Sprint Planning"
		assert c.session_lead == "emp-001"

	def test_session_duration_minutes_none_without_end(self):
		assert self._collab().session_duration_minutes is None

	def test_session_duration_minutes_computed(self):
		c = self._collab(end_time=_dt(11))
		assert c.session_duration_minutes == 60

	def test_human_ai_ratio_infinite_no_ai(self):
		c = self._collab(human_hours_contributed=Decimal("2"), ai_compute_hours=Decimal("0"))
		assert c.human_ai_ratio == float("inf")

	def test_human_ai_ratio_computed(self):
		c = self._collab(
			human_hours_contributed=Decimal("4"),
			ai_compute_hours=Decimal("2"),
		)
		assert c.human_ai_ratio == pytest.approx(2.0)

	def test_efficiency_score_bounds(self):
		with pytest.raises(ValidationError):
			self._collab(efficiency_score=1.5)

	def test_planned_duration_positive(self):
		with pytest.raises(ValidationError):
			self._collab(planned_duration_minutes=0)
