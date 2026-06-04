"""Service tests for pharma_ctr."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import pytest
from datetime import datetime

from capabilities.pharma.ctr.service import ClinicalTrialsService
from capabilities.pharma.ctr.models import (
	AdverseEventCreate, ClinicalTrialCreate, TrialPatientCreate, TrialSiteCreate,
)


def svc():
	return ClinicalTrialsService()


def test_describe():
	s = svc()
	c = s.describe("t1")
	assert c["capability"] == "pharma_ctr"


def test_create_trial():
	s = svc()
	payload = ClinicalTrialCreate(
		tenant_id="t1", trial_number="TRIAL-001", phase="phase_2",
		trial_type="interventional", title="Study of Drug X in Hypertension",
		sponsor_id="SPONSOR-001", blinding="double_blind",
		indication="hypertension", created_by="cra",
	)
	trial = s.create_trial(payload)
	assert trial.trial_number == "TRIAL-001"
	assert trial.phase == "phase_2"


def test_activate_trial_requires_irb():
	s = svc()
	payload = ClinicalTrialCreate(
		tenant_id="t1", trial_number="TRIAL-002", phase="phase_1",
		trial_type="first_in_human", title="FIH Study", sponsor_id="SP-001",
		blinding="open_label", indication="oncology", created_by="cra",
	)
	trial = s.create_trial(payload)
	with pytest.raises(PermissionError):
		s.activate_trial(trial.id, "t1", "")


def test_select_site():
	s = svc()
	trial_payload = ClinicalTrialCreate(
		tenant_id="t1", trial_number="TRIAL-003", phase="phase_3",
		trial_type="interventional", title="Phase 3 study",
		sponsor_id="SP-001", blinding="double_blind", indication="diabetes",
		created_by="cra",
	)
	trial = s.create_trial(trial_payload)
	site_payload = TrialSiteCreate(
		tenant_id="t1", trial_id=trial.id, site_number="SITE-001",
		site_name="City Hospital", country="KE",
		principal_investigator_id="PI-001", target_enrollment=30, created_by="cra",
	)
	site = s.select_site(site_payload)
	assert site.site_number == "SITE-001"
	assert site.status == "pre_selected"


def test_enrol_patient_requires_ic():
	s = svc()
	with pytest.raises(PermissionError):
		s._enforce({
			"tenant_id": "t1",
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"gcp_compliant": True,
			"operation": "enrol_patient",
			"site_initiated": True,
			"informed_consent_obtained": False,
			"eligibility_confirmed": True,
		})


def test_enrol_patient_success():
	s = svc()
	trial_payload = ClinicalTrialCreate(
		tenant_id="t1", trial_number="TRIAL-004", phase="phase_2",
		trial_type="interventional", title="Study 4",
		sponsor_id="SP-001", blinding="single_blind", indication="asthma",
		created_by="cra",
	)
	trial = s.create_trial(trial_payload)
	site_payload = TrialSiteCreate(
		tenant_id="t1", trial_id=trial.id, site_number="SITE-002",
		site_name="General Hospital", country="UG",
		principal_investigator_id="PI-002", target_enrollment=20, created_by="cra",
	)
	site = s.select_site(site_payload)
	# Manually set site to initiated for test
	from capabilities.pharma.ctr.models import TrialSite
	data = site.model_dump()
	data["status"] = "initiated"
	s._sites[(site.tenant_id, site.id)] = TrialSite(**data)

	patient_payload = TrialPatientCreate(
		tenant_id="t1", trial_id=trial.id, site_id=site.id,
		patient_code="PT-001", created_by="nurse",
	)
	patient = s.enrol_patient(patient_payload, datetime.utcnow())
	assert patient.patient_code == "PT-001"
	assert patient.status == "enrolled"


def test_report_ae():
	s = svc()
	trial_payload = ClinicalTrialCreate(
		tenant_id="t1", trial_number="TRIAL-005", phase="phase_2b",
		trial_type="interventional", title="Study 5",
		sponsor_id="SP-001", blinding="double_blind", indication="copd",
		created_by="cra",
	)
	trial = s.create_trial(trial_payload)
	ae_payload = AdverseEventCreate(
		tenant_id="t1", trial_id=trial.id, patient_id="PT-001",
		site_id="SITE-001", ae_type="adverse_event", severity_grade="grade_2",
		onset_date=datetime.utcnow(), narrative="Mild nausea",
		causality="possible", created_by="investigator",
	)
	ae = s.report_ae(ae_payload)
	assert ae.severity_grade == "grade_2"


def test_file_submission():
	s = svc()
	trial_payload = ClinicalTrialCreate(
		tenant_id="t1", trial_number="TRIAL-006", phase="phase_3",
		trial_type="interventional", title="Study 6",
		sponsor_id="SP-001", blinding="double_blind", indication="htn",
		created_by="ra",
	)
	trial = s.create_trial(trial_payload)
	sub = s.file_submission(
		"t1", trial.id, "cta", "fda", "COVER-001", "DOSSIER-001", "ra_mgr"
	)
	assert sub.authority == "fda"
	assert sub.submission_type == "cta"


def test_dashboard_summary():
	s = svc()
	trial_payload = ClinicalTrialCreate(
		tenant_id="t1", trial_number="TRIAL-007", phase="phase_2",
		trial_type="interventional", title="Study 7",
		sponsor_id="SP-001", blinding="double_blind", indication="hf",
		created_by="cra",
	)
	s.create_trial(trial_payload)
	summary = s.dashboard_summary("t1")
	assert summary["trial_count"] == 1
