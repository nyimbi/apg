"""Service tests for pharma_pvi."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import pytest
from datetime import datetime, timedelta

from capabilities.pharma.pvi.service import PharmacovigilanceService
from capabilities.pharma.pvi.models import AdvEventCaseCreate


def svc():
	return PharmacovigilanceService()


def test_describe():
	s = svc()
	c = s.describe("t1")
	assert c["capability"] == "pharma_pvi"


def test_create_case():
	s = svc()
	payload = AdvEventCaseCreate(
		tenant_id="t1", case_number="CASE-001", source="spontaneous",
		case_type="adverse_event", product_id="PROD-A",
		suspect_drug="Drug X", report_date=datetime.utcnow(), created_by="pv_analyst",
	)
	case = s.create_case(payload)
	assert case.case_number == "CASE-001"
	assert case.status == "new"


def test_process_case():
	s = svc()
	payload = AdvEventCaseCreate(
		tenant_id="t1", case_number="CASE-002", source="healthcare_professional",
		case_type="adverse_event", product_id="PROD-B",
		suspect_drug="Drug Y", report_date=datetime.utcnow(), created_by="pv",
	)
	case = s.create_case(payload)
	processed = s.process_case(
		case.id, "t1", narrative="Patient experienced rash",
		causality="possible", meddra_pt="Rash", meddra_soc="Skin disorders",
		processed_by="pv_analyst",
	)
	assert processed.status == "in_progress"
	assert processed.meddra_coded is True


def test_process_case_denied_no_meddra():
	s = svc()
	with pytest.raises(PermissionError):
		s._enforce({
			"tenant_id": "t1",
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "process_case",
			"meddra_coded": False,
			"narrative_present": True,
			"causality_assessed": True,
			"duplicate_check_done": True,
		})


def test_close_case_serious_requires_medical_review():
	s = svc()
	payload = AdvEventCaseCreate(
		tenant_id="t1", case_number="CASE-003", source="spontaneous",
		case_type="serious_adverse_event", product_id="PROD-C",
		suspect_drug="Drug Z", report_date=datetime.utcnow(),
		serious=True, created_by="pv",
	)
	case = s.create_case(payload)
	with pytest.raises(PermissionError):
		s.close_case(case.id, "t1", "resolved", medical_reviewed=False)


def test_close_case_success():
	s = svc()
	payload = AdvEventCaseCreate(
		tenant_id="t1", case_number="CASE-004", source="spontaneous",
		case_type="adverse_event", product_id="PROD-A",
		suspect_drug="Drug X", report_date=datetime.utcnow(), created_by="pv",
	)
	case = s.create_case(payload)
	closed = s.close_case(case.id, "t1", "resolved", medical_reviewed=True)
	assert closed.status == "closed_valid"


def test_create_signal():
	s = svc()
	signal = s.create_signal(
		"t1", "SIG-001", "PROD-A", "new_safety_signal",
		"Hepatotoxicity", "Elevated liver enzymes observed",
		"signal_team", "disproportionality", "analyst",
	)
	assert signal.signal_number == "SIG-001"
	assert signal.status == "new"


def test_create_psur():
	s = svc()
	now = datetime.utcnow()
	psur = s.create_psur(
		"t1", "PSUR-001", "PROD-A", "psur",
		now, now - timedelta(days=365 * 5),
		now - timedelta(days=365), now, "IBRD-001", "pv_manager",
	)
	assert psur.report_number == "PSUR-001"
	assert psur.ibrd_reference == "IBRD-001"


def test_submit_psur():
	s = svc()
	now = datetime.utcnow()
	psur = s.create_psur(
		"t1", "PSUR-002", "PROD-B", "pbrer",
		now, now - timedelta(days=365 * 3),
		now - timedelta(days=365), now, "IBRD-002", "pv",
	)
	with pytest.raises(PermissionError):
		s.submit_psur(psur.id, "t1", benefit_risk_assessed=False)
	submitted = s.submit_psur(psur.id, "t1", benefit_risk_assessed=True)
	assert submitted.status == "submitted"


def test_mark_duplicate():
	s = svc()
	p1 = AdvEventCaseCreate(tenant_id="t1", case_number="CASE-010", source="spontaneous",
							case_type="adverse_event", product_id="PA", suspect_drug="DX",
							report_date=datetime.utcnow(), created_by="pv")
	p2 = AdvEventCaseCreate(tenant_id="t1", case_number="CASE-011", source="spontaneous",
							case_type="adverse_event", product_id="PA", suspect_drug="DX",
							report_date=datetime.utcnow(), created_by="pv")
	c1 = s.create_case(p1)
	c2 = s.create_case(p2)
	dup = s.mark_duplicate(c2.id, "t1", c1.id)
	assert dup.status == "duplicate"
	assert dup.duplicate_of == c1.id


def test_dashboard_summary():
	s = svc()
	payload = AdvEventCaseCreate(
		tenant_id="t1", case_number="CASE-020", source="patient",
		case_type="adverse_event", product_id="PA", suspect_drug="DX",
		report_date=datetime.utcnow(), created_by="pv",
	)
	s.create_case(payload)
	summary = s.dashboard_summary("t1")
	assert summary["case_count"] == 1
	assert "signal_count" in summary
