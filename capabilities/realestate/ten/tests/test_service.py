"""Service tests for Tenant Management (ten)."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from capabilities.realestate.ten.service import TenService
from capabilities.realestate.ten.models import (
	TenantEntityCreate, TenantEntityUpdate, TenantType, TenantStatus,
	OnboardingStepRecord, OnboardingStep,
	ServiceRequestCreate, ServiceRequestUpdate, ServiceRequestType, RequestStatus,
	CommunicationCreate, CommunicationChannel,
	SatisfactionSurveyCreate,
	TenantScoreCreate,
	TenantEscalationCreate, EscalationType,
	CreditGrade,
)

loop = asyncio.get_event_loop()
T = "test-tenant"
MANDATORY = {"referencing", "credit_check", "deposit_registration"}


def _svc():
	return TenService()


def _tenant(svc, **kwargs):
	defaults = dict(
		tenant_id=T, name="ACME Corp", tenant_type=TenantType.corporate,
		email="acme@test.com", created_by="u",
	)
	defaults.update(kwargs)
	return loop.run_until_complete(svc.register_tenant(TenantEntityCreate(**defaults)))


def _complete_mandatory_steps(svc, tenant_entity_id):
	for step_name in MANDATORY:
		loop.run_until_complete(svc.complete_onboarding_step(OnboardingStepRecord(
			tenant_id=T, tenant_entity_id=tenant_entity_id,
			step=OnboardingStep(step_name), completed_by="u",
		)))


# ── Tenant Entity ─────────────────────────────────────────────────────────────

def test_register_tenant():
	svc = _svc()
	t = _tenant(svc)
	assert t.id
	assert t.status == TenantStatus.prospect


def test_get_tenant():
	svc = _svc()
	t = _tenant(svc)
	fetched = loop.run_until_complete(svc.get_tenant(t.id, T))
	assert fetched.name == "ACME Corp"


def test_list_tenants_by_status():
	svc = _svc()
	t1 = _tenant(svc, name="Corp1", email="c1@t.com")
	t2 = _tenant(svc, name="Corp2", email="c2@t.com")
	all_t = loop.run_until_complete(svc.list_tenants(T))
	assert len(all_t) == 2


def test_update_tenant():
	svc = _svc()
	t = _tenant(svc)
	updated = loop.run_until_complete(svc.update_tenant(t.id, T, TenantEntityUpdate(email="new@acme.com")))
	assert updated.email == "new@acme.com"


def test_blacklisted_tenant_cannot_be_activated():
	svc = _svc()
	t = _tenant(svc)
	loop.run_until_complete(svc.blacklist_tenant(t.id, T, "fraud"))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.activate_tenant(t.id, T))


# ── Onboarding ────────────────────────────────────────────────────────────────

def test_complete_onboarding_steps():
	svc = _svc()
	t = _tenant(svc)
	_complete_mandatory_steps(svc, t.id)
	progress = loop.run_until_complete(svc.get_onboarding_progress(t.id, T))
	assert progress["mandatory_complete"] is True
	assert len(progress["completed_steps"]) >= 3


def test_activate_after_onboarding():
	svc = _svc()
	t = _tenant(svc)
	_complete_mandatory_steps(svc, t.id)
	activated = loop.run_until_complete(svc.activate_tenant(t.id, T))
	assert activated.status == TenantStatus.active


def test_activation_without_mandatory_steps_raises():
	svc = _svc()
	t = _tenant(svc)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.activate_tenant(t.id, T))


# ── Service Requests ──────────────────────────────────────────────────────────

def test_raise_service_request():
	svc = _svc()
	t = _tenant(svc)
	sr = loop.run_until_complete(svc.raise_service_request(ServiceRequestCreate(
		tenant_id=T, tenant_entity_id=t.id, property_id="prop-1",
		request_type=ServiceRequestType.maintenance_request,
		subject="AC not working", description="AC unit in server room failed",
		created_by="u",
	)))
	assert sr.ref.startswith("SR-")
	assert sr.sla_response_deadline is not None


def test_resolve_service_request():
	svc = _svc()
	t = _tenant(svc)
	sr = loop.run_until_complete(svc.raise_service_request(ServiceRequestCreate(
		tenant_id=T, tenant_entity_id=t.id, property_id="prop-1",
		request_type=ServiceRequestType.general_enquiry,
		subject="Parking query", description="Visitor parking allocation",
		created_by="u",
	)))
	resolved = loop.run_until_complete(svc.resolve_service_request(sr.id, T, "Parking bay P-12 allocated", 5))
	assert resolved.status == RequestStatus.resolved
	assert resolved.satisfaction_rating == 5


def test_list_service_requests_by_status():
	svc = _svc()
	t = _tenant(svc)
	for i in range(3):
		loop.run_until_complete(svc.raise_service_request(ServiceRequestCreate(
			tenant_id=T, tenant_entity_id=t.id, property_id="p1",
			request_type=ServiceRequestType.general_enquiry,
			subject=f"Query {i}", description="desc", created_by="u",
		)))
	open_reqs = loop.run_until_complete(svc.list_service_requests(T, status="open"))
	assert len(open_reqs) == 3


# ── Communication ─────────────────────────────────────────────────────────────

def test_send_communication():
	svc = _svc()
	t = _tenant(svc)
	comm = loop.run_until_complete(svc.send_communication(CommunicationCreate(
		tenant_id=T, tenant_entity_id=t.id, channel=CommunicationChannel.email,
		subject="Welcome", body="Welcome to the building.",
		sent_by="property_manager", created_by="u",
	)))
	assert comm.delivered is True


# ── Satisfaction ──────────────────────────────────────────────────────────────

def test_satisfaction_survey():
	svc = _svc()
	t = _tenant(svc)
	survey = loop.run_until_complete(svc.record_satisfaction_survey(SatisfactionSurveyCreate(
		tenant_id=T, tenant_entity_id=t.id, property_id="prop-1",
		survey_period="2025-Q1",
		ratings={"maintenance_quality": 4, "response_time": 3, "overall": 4},
		created_by="u",
	)))
	assert survey.average_score > 0
	assert survey.score_below_threshold is False


def test_low_satisfaction_triggers_review():
	svc = _svc()
	t = _tenant(svc)
	survey = loop.run_until_complete(svc.record_satisfaction_survey(SatisfactionSurveyCreate(
		tenant_id=T, tenant_entity_id=t.id, property_id="prop-1",
		survey_period="2025-Q2",
		ratings={"overall": 2, "communication": 1},
		created_by="u",
	)))
	assert survey.review_triggered is True


def test_satisfaction_trend():
	svc = _svc()
	t = _tenant(svc)
	loop.run_until_complete(svc.record_satisfaction_survey(SatisfactionSurveyCreate(
		tenant_id=T, tenant_entity_id=t.id, property_id="p1",
		survey_period="2024-Q4", ratings={"overall": 3}, created_by="u",
	)))
	loop.run_until_complete(svc.record_satisfaction_survey(SatisfactionSurveyCreate(
		tenant_id=T, tenant_entity_id=t.id, property_id="p1",
		survey_period="2025-Q1", ratings={"overall": 5}, created_by="u",
	)))
	trend = loop.run_until_complete(svc.get_satisfaction_trend(T, t.id))
	assert trend["trend"] == "improving"


# ── Tenant Scoring ────────────────────────────────────────────────────────────

def test_calculate_score():
	svc = _svc()
	t = _tenant(svc)
	score = loop.run_until_complete(svc.calculate_tenant_score(TenantScoreCreate(
		tenant_id=T, tenant_entity_id=t.id,
		model="payment_history", score=Decimal("75"),
		scored_by="system",
	)))
	assert score.retention_risk_flagged is False


def test_low_score_flags_retention_risk():
	svc = _svc()
	t = _tenant(svc)
	score = loop.run_until_complete(svc.calculate_tenant_score(TenantScoreCreate(
		tenant_id=T, tenant_entity_id=t.id,
		model="lease_compliance", score=Decimal("30"),
		scored_by="system",
	)))
	assert score.retention_risk_flagged is True


def test_assign_credit_grade():
	svc = _svc()
	t = _tenant(svc)
	updated = loop.run_until_complete(svc.assign_credit_grade(t.id, T, CreditGrade.A))
	assert updated.credit_grade == CreditGrade.A


# ── Escalation ────────────────────────────────────────────────────────────────

def test_raise_and_resolve_escalation():
	svc = _svc()
	t = _tenant(svc)
	esc = loop.run_until_complete(svc.raise_escalation(TenantEscalationCreate(
		tenant_id=T, tenant_entity_id=t.id,
		escalation_type=EscalationType.noise_complaint,
		description="Loud music after hours", created_by="u",
	)))
	assert esc.status == "open"
	resolved = loop.run_until_complete(svc.resolve_escalation(esc.id, T, "Formal warning issued"))
	assert resolved.status == "resolved"


# ── Retention at Risk ─────────────────────────────────────────────────────────

def test_retention_at_risk_list():
	svc = _svc()
	t = _tenant(svc)
	_complete_mandatory_steps(svc, t.id)
	loop.run_until_complete(svc.activate_tenant(t.id, T))
	loop.run_until_complete(svc.calculate_tenant_score(TenantScoreCreate(
		tenant_id=T, tenant_entity_id=t.id, model="payment_history",
		score=Decimal("25"), scored_by="system",
	)))
	at_risk = loop.run_until_complete(svc.get_retention_at_risk(T))
	assert any(x.id == t.id for x in at_risk)
