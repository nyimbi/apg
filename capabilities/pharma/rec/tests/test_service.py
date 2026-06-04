"""Service tests for pharma_rec."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import pytest
from datetime import datetime, timedelta

from capabilities.pharma.rec.service import RegulatoryComplianceService


def svc():
	return RegulatoryComplianceService()


def test_describe():
	s = svc()
	c = s.describe("t1")
	assert c["capability"] == "pharma_rec"


def test_register_compliance():
	s = svc()
	record = s.register_compliance(
		"t1", "eu_gmp", "EU GMP Requirements",
		["SITE-KE-01"], "qa_director", "qa_director",
	)
	assert record.framework == "eu_gmp"
	assert record.status == "active"


def test_record_inspection():
	s = svc()
	inspection = s.record_inspection(
		"t1", "INSP-001", "gmp_inspection", "EMA", "SITE-KE-01",
		True, "qa_mgr",
	)
	assert inspection.inspection_number == "INSP-001"
	assert inspection.status == "planned"


def test_record_inspection_outcome():
	s = svc()
	inspection = s.record_inspection(
		"t1", "INSP-002", "gmp_inspection", "FDA", "SITE-US-01",
		False, "qa",
	)
	result = s.record_inspection_outcome(inspection.id, "t1", "voluntary_action_indicated", 3)
	assert result.outcome == "voluntary_action_indicated"
	assert result.status == "completed"


def test_warning_letter_deadline_set():
	s = svc()
	inspection = s.record_inspection(
		"t1", "INSP-003", "fda_inspection", "FDA", "SITE-US-02", False, "qa",
	)
	result = s.record_inspection_outcome(inspection.id, "t1", "warning_letter", 5)
	assert result.response_deadline is not None
	delta = result.response_deadline - datetime.utcnow()
	assert 25 <= delta.days <= 31


def test_create_label():
	s = svc()
	label = s.create_label(
		"t1", "LBL-001", "PROD-A", "EU", "en",
		"1.0", "labeling_change_prior_approval", "ra_mgr",
	)
	assert label.label_number == "LBL-001"
	assert label.status == "draft"


def test_approve_label():
	s = svc()
	label = s.create_label(
		"t1", "LBL-002", "PROD-B", "US", "en",
		"2.0", "annual_report_labeling", "ra",
	)
	approved = s.approve_label(label.id, "t1", "QP-001")
	assert approved.qp_approved is True
	assert approved.status == "approved"


def test_create_pms():
	s = svc()
	pms = s.create_pms("t1", "PMS-001", "PROD-A", "post_market_surveillance", "ra_mgr")
	assert pms.pms_number == "PMS-001"
	assert pms.status == "planned"


def test_record_intel():
	s = svc()
	intel = s.record_intel(
		"t1", "INTEL-001", "guidance_document", "us_fda",
		"New ICH Q12 Guidance", "Updated guidance on lifecycle management",
		"ra_analyst",
	)
	assert intel.intel_number == "INTEL-001"
	assert intel.impact_assessed is False


def test_create_commitment():
	s = svc()
	commitment = s.create_commitment(
		"t1", "CMT-001", "PROD-A", "EMA",
		"Submit paediatric study report",
		datetime.utcnow() + timedelta(days=365),
		[{"milestone": "Protocol", "due": "Q1"}],
		"ra_mgr",
	)
	assert commitment.commitment_number == "CMT-001"
	assert commitment.status == "open"


def test_overdue_commitments():
	s = svc()
	s.create_commitment(
		"t1", "CMT-002", "PROD-B", "FDA",
		"Submit post-approval study",
		datetime.utcnow() - timedelta(days=1),
		[{"milestone": "Report"}], "ra",
	)
	overdue = s.check_overdue_commitments("t1")
	assert len(overdue) == 1
	assert overdue[0].overdue is True


def test_dashboard_summary():
	s = svc()
	s.register_compliance("t1", "eu_gmp", "EU GMP", ["SITE-001"], "qa", "qa")
	summary = s.dashboard_summary("t1")
	assert summary["framework_count"] == 1
	assert "open_inspections" in summary
