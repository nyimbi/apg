"""Service tests for pharma_qms."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import pytest
from datetime import datetime, timedelta

from capabilities.pharma.qms.service import QualityManagementService
from capabilities.pharma.qms.models import CapaCreate, ChangeControlCreate


def svc():
	return QualityManagementService()


def test_describe():
	s = svc()
	c = s.describe("t1")
	assert c["capability"] == "pharma_qms"


def test_initiate_change():
	s = svc()
	payload = ChangeControlCreate(
		tenant_id="t1", change_number="CHG-001", title="Process Change",
		change_type="major", description="Change to process X",
		raised_by="qa_mgr", created_by="qa_mgr",
	)
	change = s.initiate_change(payload)
	assert change.change_number == "CHG-001"
	assert change.status == "draft"


def test_approve_change():
	s = svc()
	payload = ChangeControlCreate(
		tenant_id="t1", change_number="CHG-002", title="Minor update",
		change_type="minor", description="update label", raised_by="qa",
		created_by="qa",
	)
	change = s.initiate_change(payload)
	approved = s.approve_change(change.id, "t1", "APR-001", impact_assessed=True, risk_assessed=True)
	assert approved.status == "approved"


def test_approve_change_denied_no_impact():
	s = svc()
	with pytest.raises(PermissionError):
		s._enforce({
			"tenant_id": "t1",
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "approve_change",
			"impact_assessed": False,
			"risk_assessed": True,
		})


def test_create_capa():
	s = svc()
	payload = CapaCreate(
		tenant_id="t1", capa_number="CAPA-001", capa_type="corrective_action",
		title="Fix process deviation", description="Root cause: procedure not followed",
		source_reference="DEV-001", owner_id="qa_lead",
		target_completion_date=datetime.utcnow() + timedelta(days=30),
		created_by="qa_lead",
	)
	capa = s.create_capa(payload)
	assert capa.capa_number == "CAPA-001"
	assert capa.status == "open"


def test_close_capa():
	s = svc()
	payload = CapaCreate(
		tenant_id="t1", capa_number="CAPA-002", capa_type="corrective_action",
		title="Fix", description="Fix deviation", source_reference="DEV-002",
		owner_id="qa", created_by="qa",
	)
	capa = s.create_capa(payload)
	closed = s.close_capa(
		capa.id, "t1", root_cause="Operator error",
		root_cause_method="5_why", effectiveness_checked=True,
		effectiveness_result="effective",
	)
	assert closed.status == "closed_effective"


def test_raise_deviation():
	s = svc()
	dev = s.raise_deviation("t1", "DEV-001", "process_deviation", "major",
							"Batch temperature exceeded spec", "operator")
	assert dev.deviation_number == "DEV-001"
	assert dev.status == "open"


def test_close_deviation():
	s = svc()
	dev = s.raise_deviation("t1", "DEV-002", "equipment_deviation", "minor",
							"Equipment calibration drift", "tech")
	closed = s.close_deviation(dev.id, "t1", "Calibration procedure updated")
	assert closed.status == "closed"
	assert closed.root_cause == "Calibration procedure updated"


def test_create_document():
	s = svc()
	doc = s.create_document("t1", "SOP-001", "Cleaning Procedure", "sop", "1.0",
							"Manufacturing", "qa_mgr", "qa_mgr")
	assert doc.document_number == "SOP-001"
	assert doc.status == "draft"


def test_approve_document():
	s = svc()
	doc = s.create_document("t1", "SOP-002", "Batch Release", "sop", "1.0",
							"QA", "qa_mgr", "qa_mgr")
	approved = s.approve_document(doc.id, "t1", "approver_id")
	assert approved.status == "effective"


def test_create_audit():
	s = svc()
	audit = s.create_audit("t1", "AUD-001", "internal", "Manufacturing",
						["auditor1"], "Full GMP audit", "qa_mgr")
	assert audit.audit_number == "AUD-001"
	assert audit.status == "planned"


def test_close_audit_with_findings_and_capa():
	s = svc()
	audit = s.create_audit("t1", "AUD-002", "internal", "Warehouse",
						["auditor1"], "GDP audit", "qa")
	closed = s.close_audit(audit.id, "t1", "RPT-001", findings_count=2,
						capa_references=["CAPA-001", "CAPA-002"])
	assert closed.status == "closed"
	assert closed.findings_count == 2


def test_dashboard_summary():
	s = svc()
	payload = ChangeControlCreate(
		tenant_id="t1", change_number="CHG-003", title="Test",
		change_type="minor", description="test", raised_by="qa", created_by="qa",
	)
	s.initiate_change(payload)
	summary = s.dashboard_summary("t1")
	assert summary["open_changes"] >= 1
	assert "overdue_capas" in summary
