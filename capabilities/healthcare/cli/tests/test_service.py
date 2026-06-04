"""Tests for ClinicalManagementService."""

from __future__ import annotations
import asyncio, sys, os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cli.models import CarePlanCreate, CDSAlertCreate, ClinicalWorkflowCreate, HandoffCreate, ProtocolCreate
from cli.service import ClinicalManagementService, PolicyViolationError


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def svc():
	return ClinicalManagementService()


def make_cp(s, tid="t", with_team=True) -> any:
	return run(s.create_care_plan(CarePlanCreate(
		tenant_id=tid, patient_id="p1", title="Diabetes Care Plan",
		description="Manage T2DM", goals=["HbA1c < 7%"],
		care_team_ids=["dr1", "nurse1"] if with_team else [],
		created_by="dr1",
	)))


def test_create_care_plan():
	s = svc()
	cp = make_cp(s)
	assert cp.id and cp.status == "draft"


def test_activate_care_plan_with_team():
	s = svc()
	cp = make_cp(s, with_team=True)
	activated = run(s.activate_care_plan("t", cp.id))
	assert activated.status == "active"


def test_activate_care_plan_no_team_denied():
	s = svc()
	cp = make_cp(s, with_team=False)
	try:
		run(s.activate_care_plan("t", cp.id))
		assert False
	except PolicyViolationError:
		pass


def test_complete_care_plan():
	s = svc()
	cp = make_cp(s)
	run(s.activate_care_plan("t", cp.id))
	completed = run(s.complete_care_plan("t", cp.id))
	assert completed.status == "completed"


def test_add_intervention():
	s = svc()
	cp = make_cp(s)
	updated = run(s.add_intervention("t", cp.id, "medication", "Start Metformin 500mg BID"))
	assert len(updated.interventions) == 1
	assert updated.interventions[0]["type"] == "medication"


def test_add_invalid_intervention_denied():
	s = svc()
	cp = make_cp(s)
	try:
		run(s.add_intervention("t", cp.id, "unknown_type", "desc"))
		assert False
	except PolicyViolationError:
		pass


def test_create_protocol():
	s = svc()
	proto = run(s.create_protocol(ProtocolCreate(
		tenant_id="t", protocol_type="sepsis_bundle", name="Sepsis Bundle",
		description="3-hour bundle", activation_criteria="Lactate >= 2 mmol/L",
		steps=[{"step": 1, "action": "Blood cultures"}],
		evidence_reference="Surviving Sepsis Campaign 2021", created_by="dr1",
	)))
	assert proto.status == "active"


def test_create_protocol_unsupported_type_denied():
	s = svc()
	try:
		run(s.create_protocol(ProtocolCreate(
			tenant_id="t", protocol_type="unknown_proto", name="X",
			description="X", activation_criteria="X",
			evidence_reference="src", created_by="dr1",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_create_and_transition_workflow():
	s = svc()
	wf = run(s.create_workflow(ClinicalWorkflowCreate(
		tenant_id="t", patient_id="p1", title="Draw HbA1c",
		description="Lab order for HbA1c", assigned_to="nurse1",
		due_at=datetime.utcnow() + timedelta(hours=4), created_by="dr1",
	)))
	assert wf.state == "pending"
	updated = run(s.transition_workflow("t", wf.id, "in_progress"))
	assert updated.state == "in_progress"
	completed = run(s.transition_workflow("t", wf.id, "completed"))
	assert completed.state == "completed" and completed.completed_at is not None


def test_create_cds_alert():
	s = svc()
	alert = run(s.create_cds_alert(CDSAlertCreate(
		tenant_id="t", patient_id="p1", cds_type="sepsis_screening",
		priority="critical", message="Sepsis screening positive",
		evidence_reference="qSOFA >= 2", suggested_action="Initiate sepsis bundle",
		created_by="system",
	)))
	assert alert.priority == "critical" and alert.status == "active"


def test_acknowledge_cds_alert():
	s = svc()
	alert = run(s.create_cds_alert(CDSAlertCreate(tenant_id="t", patient_id="p1", cds_type="deterioration_alert", priority="high", message="NEWS2 >= 7", evidence_reference="NEWS2", suggested_action="Notify senior", created_by="system")))
	ack = run(s.acknowledge_cds_alert("t", alert.id, "dr1"))
	assert ack.status == "acknowledged" and ack.acknowledged_by == "dr1"


def test_record_handoff():
	s = svc()
	handoff = run(s.record_handoff(HandoffCreate(
		tenant_id="t", patient_id="p1", handoff_type="shift_change",
		from_provider_id="dr_day", to_provider_id="dr_night",
		situation="Pt admitted with DKA", background="T2DM HbA1c 11%",
		assessment="Improving, on insulin drip", recommendation="Continue drip, recheck BMP in 2h",
		structured_format_used=True, created_by="dr_day",
	)))
	assert handoff.id and handoff.acknowledged_by is None


def test_handoff_no_structured_format_denied():
	s = svc()
	try:
		run(s.record_handoff(HandoffCreate(
			tenant_id="t", patient_id="p1", handoff_type="shift_change",
			from_provider_id="dr1", to_provider_id="dr2",
			situation="S", background="B", assessment="A", recommendation="R",
			structured_format_used=False, created_by="dr1",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_acknowledge_handoff():
	s = svc()
	handoff = run(s.record_handoff(HandoffCreate(
		tenant_id="t", patient_id="p1", handoff_type="transfer",
		from_provider_id="er_dr", to_provider_id="icu_dr",
		situation="S", background="B", assessment="A", recommendation="R",
		structured_format_used=True, created_by="er_dr",
	)))
	ack = run(s.acknowledge_handoff("t", handoff.id, "icu_dr"))
	assert ack.acknowledged_by == "icu_dr"


def test_dashboard_summary():
	s = svc()
	make_cp(s)
	summary = run(s.dashboard_summary("t"))
	assert summary["care_plans"]["total"] == 1
	assert "workflows" in summary and "cds_alerts" in summary
