"""Service layer tests for pharma_com."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import pytest
from datetime import datetime

from capabilities.pharma.com.service import CommercialOperationsService
from capabilities.pharma.com.models import (
	CallRecordCreate, CommercialPlanCreate, HcpInteractionCreate,
	SalesRepCreate, SampleDispensingCreate, TerritoryCreate, TerritoryUpdate,
)


def svc():
	return CommercialOperationsService()


def test_describe_returns_contract():
	s = svc()
	contract = s.describe("test_tenant")
	assert contract["capability"] == "pharma_com"
	assert contract["configuration"]["tenant_id"] == "test_tenant"


def test_create_territory_happy_path():
	s = svc()
	payload = TerritoryCreate(
		tenant_id="t1", territory_type="regional", name="East Region",
		owner_id="rep_01", product_ids=["prod_a"], approval_reference="APR-001",
		created_by="admin",
	)
	territory = s.create_territory(payload)
	assert territory.name == "East Region"
	assert territory.territory_type == "regional"
	assert territory.tenant_id == "t1"


def test_create_territory_cross_tenant_isolation():
	s = svc()
	p1 = TerritoryCreate(tenant_id="t1", territory_type="national", name="T1 Territory",
						owner_id="r1", approval_reference="A1", created_by="admin")
	p2 = TerritoryCreate(tenant_id="t2", territory_type="regional", name="T2 Territory",
						owner_id="r2", approval_reference="A2", created_by="admin")
	s.create_territory(p1)
	s.create_territory(p2)
	assert len(s.list_territories("t1")) == 1
	assert len(s.list_territories("t2")) == 1


def test_update_territory():
	s = svc()
	payload = TerritoryCreate(
		tenant_id="t1", territory_type="regional", name="Old Name",
		owner_id="rep_01", approval_reference="APR-001", created_by="admin",
	)
	territory = s.create_territory(payload)
	updated = s.update_territory(territory.id, "t1", TerritoryUpdate(name="New Name"))
	assert updated.name == "New Name"


def test_assign_rep_happy_path():
	s = svc()
	p = TerritoryCreate(tenant_id="t1", territory_type="regional", name="R1",
						owner_id="mgr", approval_reference="A1", created_by="admin")
	territory = s.create_territory(p)
	rep_payload = SalesRepCreate(
		tenant_id="t1", rep_type="primary_care", employee_id="EMP001",
		name="Jane Doe", territory_id=territory.id, quota=100000.0,
		certification_reference="CERT-001", created_by="admin",
	)
	rep = s.assign_rep(rep_payload)
	assert rep.name == "Jane Doe"
	assert rep.territory_id == territory.id


def test_record_call_happy_path():
	s = svc()
	payload = CallRecordCreate(
		tenant_id="t1", rep_id="rep1", physician_id="doc1",
		call_type="detailing", products_discussed=["prod_a"],
		outcome="interested", call_date=datetime.utcnow(), created_by="rep1",
	)
	call = s.record_call(payload)
	assert call.physician_id == "doc1"
	assert call.call_type == "detailing"


def test_record_call_denied_no_product():
	s = svc()
	with pytest.raises(PermissionError):
		s._enforce({
			"tenant_id": "t1",
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_call",
			"physician_id_present": True,
			"call_type_supported": True,
			"product_present": False,
		})


def test_dispense_sample_pdma_enforced():
	s = svc()
	with pytest.raises(PermissionError):
		payload = SampleDispensingCreate(
			tenant_id="t1", rep_id="rep1", physician_id="doc1",
			sample_type="promotional_sample", product_id="prod_a",
			lot_number="LOT001", expiry_date="2027-01-01", quantity=2,
			hcp_signature_reference="SIG001", pdma_compliant=False,
			created_by="rep1",
		)
		s.dispense_sample(payload)


def test_dispense_sample_success():
	s = svc()
	payload = SampleDispensingCreate(
		tenant_id="t1", rep_id="rep1", physician_id="doc1",
		sample_type="promotional_sample", product_id="prod_a",
		lot_number="LOT001", expiry_date="2027-01-01", quantity=2,
		hcp_signature_reference="SIG001", pdma_compliant=True,
		created_by="rep1",
	)
	sample = s.dispense_sample(payload)
	assert sample.lot_number == "LOT001"
	assert sample.pdma_compliant is True


def test_record_interaction():
	s = svc()
	payload = HcpInteractionCreate(
		tenant_id="t1", rep_id="rep1", hcp_id="doc1",
		interaction_type="office_visit", products_discussed=["prod_a"],
		spend_amount=15.0, interaction_date=datetime.utcnow(),
		created_by="rep1",
	)
	interaction = s.record_interaction(payload)
	assert interaction.hcp_id == "doc1"
	assert interaction.spend_amount == 15.0


def test_create_plan_and_approve():
	s = svc()
	payload = CommercialPlanCreate(
		tenant_id="t1", plan_name="Q1 2026 Plan", plan_period="Q1-2026",
		territory_ids=["terr1"], product_ids=["prod_a"], total_quota=500000.0,
		created_by="admin",
	)
	plan = s.create_plan(payload)
	assert plan.status == "draft"
	approved = s.approve_plan(plan.id, "t1", "APR-PLAN-001")
	assert approved.status == "approved"
	assert approved.approval_reference == "APR-PLAN-001"


def test_aggregate_spend_cap_enforced():
	s = svc()
	with pytest.raises(PermissionError):
		s._enforce({
			"tenant_id": "t1",
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_spend",
			"amount_above_threshold": False,
			"receipt_present": True,
			"amount_above_approval_threshold": False,
			"pre_approval_present": True,
			"aggregate_cap_exceeded": True,
		})


def test_dashboard_summary():
	s = svc()
	payload = TerritoryCreate(tenant_id="t1", territory_type="regional", name="R1",
							owner_id="mgr", approval_reference="A1", created_by="admin")
	s.create_territory(payload)
	summary = s.dashboard_summary("t1")
	assert summary["territory_count"] == 1
	assert "rep_count" in summary
	assert summary["tenant_id"] == "t1"


def test_audit_events_recorded():
	s = svc()
	payload = TerritoryCreate(tenant_id="t1", territory_type="regional", name="R1",
							owner_id="mgr", approval_reference="A1", created_by="admin")
	s.create_territory(payload)
	assert any(e["event_type"] == "territory_created" for e in s._audit_events)


def test_list_targets_by_territory():
	s = svc()
	s.set_target("t1", "doc1", "tier_1", "terr1", ["prod_a"], 4, "SEG-001", "admin")
	s.set_target("t1", "doc2", "tier_2", "terr2", ["prod_a"], 2, "SEG-002", "admin")
	terr1_targets = s.list_targets("t1", territory_id="terr1")
	assert len(terr1_targets) == 1
	assert terr1_targets[0].physician_id == "doc1"
