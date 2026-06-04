"""Tests for SafService."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from capabilities.mining.saf.models import (
	ControlMeasureCreate,
	ControlType,
	CorrectiveActionCreate,
	HazardCategory,
	HazardCreate,
	IncidentCreate,
	IncidentType,
	LikelihoodLevel,
	ConsequenceLevel,
	PermitToWorkCreate,
	PTWType,
	ReviewStatus,
	RiskRating,
	RiskRegisterEntryCreate,
)
from capabilities.mining.saf.service import SafService

TENANT = "test_mine_saf"


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def make_service():
	return SafService(tenant_id=TENANT)


def make_incident(incident_type: IncidentType = IncidentType.NEAR_MISS) -> IncidentCreate:
	return IncidentCreate(
		tenant_id=TENANT,
		incident_type=incident_type,
		occurred_at=datetime(2026, 1, 10, 8, 30),
		location="Pit bench 5, North wall",
		mine_area="open_pit_north",
		description="Near miss with haul truck",
		reported_by="worker_001",
		supervisor_id="super_001",
	)


def test_report_incident():
	svc = make_service()
	result = run(svc.report_incident(make_incident(), created_by="worker_001"))
	assert result.incident_type == IncidentType.NEAR_MISS
	assert result.tenant_id == TENANT


def test_close_near_miss_without_investigation():
	svc = make_service()
	incident = run(svc.report_incident(make_incident(), created_by="worker_001"))
	# Near miss doesn't require investigation
	closed = run(svc.close_incident(incident.id, "Issue resolved", "super_001"))
	assert closed.status == ReviewStatus.CLOSED


def test_close_lti_requires_investigation():
	svc = make_service()
	incident = run(svc.report_incident(make_incident(IncidentType.LOST_TIME_INJURY), created_by="worker_001"))
	with pytest.raises(PermissionError, match="Investigation required"):
		run(svc.close_incident(incident.id, "Closed", "super_001"))


def test_open_investigation_then_close():
	svc = make_service()
	incident = run(svc.report_incident(make_incident(IncidentType.LOST_TIME_INJURY), created_by="worker_001"))
	run(svc.open_investigation(incident.id, "INV-2026-001"))
	closed = run(svc.close_incident(incident.id, "Investigation complete", "super_001"))
	assert closed.status == ReviewStatus.CLOSED


def test_identify_hazard_no_extreme():
	svc = make_service()
	hazard = HazardCreate(
		tenant_id=TENANT,
		hazard_category=HazardCategory.MECHANICAL,
		location="Crusher feed area",
		mine_area="process_plant",
		description="Unguarded rotating machinery",
		potential_consequence=ConsequenceLevel.MAJOR,
		likelihood=LikelihoodLevel.POSSIBLE,
		inherent_risk_rating=RiskRating.HIGH,
		control_measures=[
			ControlMeasureCreate(
				control_type=ControlType.ENGINEERING,
				description="Install machine guard",
				responsible_person_id="maint_001",
			)
		],
		identified_by="safety_officer",
		identified_at=datetime(2026, 1, 12),
	)
	result = run(svc.identify_hazard(hazard, created_by="safety_officer"))
	assert result.inherent_risk_rating == RiskRating.HIGH


def test_extreme_hazard_requires_stop_work():
	svc = make_service()
	hazard = HazardCreate(
		tenant_id=TENANT,
		hazard_category=HazardCategory.GROUND_INSTABILITY,
		location="Underground heading",
		mine_area="underground",
		description="Ground failure risk",
		potential_consequence=ConsequenceLevel.CATASTROPHIC,
		likelihood=LikelihoodLevel.LIKELY,
		inherent_risk_rating=RiskRating.EXTREME,
		identified_by="safety_officer",
		identified_at=datetime(2026, 1, 13),
		stop_work_invoked=False,  # Not invoked — should fail
	)
	with pytest.raises(PermissionError, match="extreme risk"):
		run(svc.identify_hazard(hazard, created_by="safety_officer"))


def test_issue_and_close_permit():
	svc = make_service()
	ptw = PermitToWorkCreate(
		tenant_id=TENANT,
		ptw_type=PTWType.HOT_WORK,
		location="Workshop bay 3",
		mine_area="workshop",
		work_description="Welding on conveyor frame",
		valid_from=datetime(2026, 1, 15, 7, 0),
		valid_to=datetime(2026, 1, 15, 16, 0),
		issuer_id="safety_officer_001",
	)
	permit = run(svc.issue_permit(ptw, created_by="safety_officer_001"))
	assert permit.ptw_type == PTWType.HOT_WORK
	closed = run(svc.close_permit(permit.id, "safety_officer_001"))
	assert closed.status == ReviewStatus.CLOSED


def test_permit_valid_check():
	svc = make_service()
	now = datetime.utcnow()
	ptw = PermitToWorkCreate(
		tenant_id=TENANT,
		ptw_type=PTWType.CONFINED_SPACE_ENTRY,
		location="Sump access",
		mine_area="process_plant",
		work_description="Inspect sump",
		valid_from=now - timedelta(hours=1),
		valid_to=now + timedelta(hours=5),
		issuer_id="safety_officer_001",
	)
	permit = run(svc.issue_permit(ptw, created_by="safety_officer_001"))
	assert run(svc.check_permit_valid(permit.id))


def test_corrective_action_lifecycle():
	svc = make_service()
	ca = CorrectiveActionCreate(
		tenant_id=TENANT,
		source_type="incident",
		source_id="INC-001",
		description="Install safety guard on crusher",
		assignee_id="maint_001",
		due_date=datetime(2026, 2, 1),
		priority="high",
	)
	result = run(svc.create_corrective_action(ca, created_by="safety_officer"))
	assert result.status.value == "open"
	closed = run(svc.close_corrective_action(result.id, "maint_001", "Guard installed"))
	assert closed.status.value == "closed"


def test_flag_overdue_corrective_actions():
	svc = make_service()
	ca = CorrectiveActionCreate(
		tenant_id=TENANT,
		source_type="hazard",
		source_id="HAZ-001",
		description="Replace worn hose",
		assignee_id="maint_002",
		due_date=datetime(2020, 1, 1),  # Past due
		priority="medium",
	)
	run(svc.create_corrective_action(ca, created_by="safety_officer"))
	overdue = run(svc.flag_overdue_corrective_actions())
	assert len(overdue) >= 1


def test_safety_statistics():
	svc = make_service()
	run(svc.report_incident(make_incident(IncidentType.NEAR_MISS), created_by="w1"))
	run(svc.report_incident(make_incident(IncidentType.LOST_TIME_INJURY), created_by="w2"))
	stats = run(svc.get_safety_statistics())
	assert stats["total_incidents"] == 2
	assert stats["lost_time_injuries"] == 1
	assert stats["near_misses"] == 1
