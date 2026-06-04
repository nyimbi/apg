"""Tests for PharmacyManagementService."""

from __future__ import annotations
import asyncio, sys, os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic import ValidationError

from pha.models import (
	ControlledSubstanceLogCreate, DispenseOrderCreate, DrugCreate,
	DrugInteractionCreate, InventoryItemCreate, PriorAuthCreate,
)
from pha.service import PharmacyManagementService, PolicyViolationError


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def svc():
	return PharmacyManagementService()


def make_drug(s, tid="t") -> any:
	return run(s.add_drug_to_formulary(DrugCreate(
		tenant_id=tid, drug_name="Metformin", generic_name="metformin hydrochloride",
		ndc_code="0093-7267-01", drug_type="generic", drug_schedule="non_controlled",
		dosage_form="tablet", strength="500", unit="mg", manufacturer="Teva",
		formulary_status="preferred", created_by="pharmacist1",
	)))


def test_add_drug_to_formulary():
	s = svc()
	drug = make_drug(s)
	assert drug.id and drug.drug_name == "Metformin"


def test_add_drug_invalid_type_denied():
	"""Pydantic rejects unknown enum values before the service layer sees them —
	either ValidationError (model construction) or PolicyViolationError (service gate)
	is an acceptable failure mode here."""
	s = svc()
	try:
		run(s.add_drug_to_formulary(DrugCreate(
			tenant_id="t", drug_name="X", generic_name="x", ndc_code="123",
			drug_type="unknown_type", drug_schedule="non_controlled", dosage_form="tablet",
			strength="10", unit="mg", manufacturer="M", created_by="u",
		)))
		assert False, "expected rejection of unknown drug_type"
	except (PolicyViolationError, ValidationError):
		pass


def test_mark_drug_lasa():
	s = svc()
	drug = make_drug(s)
	updated = run(s.mark_drug_lasa("t", drug.id, "Metoprolol", "look_alike"))
	assert updated.is_lasa and updated.lasa_pair == "Metoprolol"


def test_create_dispense_order():
	s = svc()
	drug = make_drug(s)
	order = run(s.create_dispense_order(DispenseOrderCreate(
		tenant_id="t", patient_id="p1", drug_id=drug.id, prescription_id="rx1",
		quantity=30.0, unit="tablets", pharmacist_verified=True,
		drug_inventory_status="in_stock", formulary_status="preferred",
		prior_auth_approved=True, step_therapy_completed=True, created_by="pharmacist1",
	)))
	assert order.status == "pending"


def test_dispense_without_pharmacist_denied():
	s = svc()
	drug = make_drug(s)
	try:
		run(s.create_dispense_order(DispenseOrderCreate(
			tenant_id="t", patient_id="p1", drug_id=drug.id, prescription_id="rx1",
			quantity=30.0, unit="tablets", pharmacist_verified=False,
			drug_inventory_status="in_stock", formulary_status="preferred",
			prior_auth_approved=True, step_therapy_completed=True, created_by="u",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_dispense_contraindicated_denied():
	s = svc()
	drug = make_drug(s)
	try:
		run(s.create_dispense_order(DispenseOrderCreate(
			tenant_id="t", patient_id="p1", drug_id=drug.id, prescription_id="rx1",
			quantity=30.0, unit="tablets", pharmacist_verified=True,
			interaction_severity="contraindicated",
			drug_inventory_status="in_stock", formulary_status="preferred",
			prior_auth_approved=True, step_therapy_completed=True, created_by="u",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_verify_and_dispense_flow():
	s = svc()
	drug = make_drug(s)
	order = run(s.create_dispense_order(DispenseOrderCreate(
		tenant_id="t", patient_id="p1", drug_id=drug.id, prescription_id="rx1",
		quantity=30.0, unit="tablets", pharmacist_verified=True,
		drug_inventory_status="in_stock", formulary_status="preferred",
		prior_auth_approved=True, step_therapy_completed=True, created_by="u",
	)))
	verified = run(s.verify_dispense("t", order.id, "pharmacist_rph1"))
	assert verified.status == "verified" and verified.pharmacist_id == "pharmacist_rph1"
	dispensed = run(s.dispense("t", order.id))
	assert dispensed.status == "dispensed" and dispensed.dispensed_at is not None


def test_record_interaction():
	s = svc()
	drug_a = make_drug(s)
	drug_b = run(s.add_drug_to_formulary(DrugCreate(
		tenant_id="t", drug_name="Warfarin", generic_name="warfarin sodium",
		ndc_code="0056-0172-70", drug_type="brand", drug_schedule="non_controlled",
		dosage_form="tablet", strength="5", unit="mg", manufacturer="BMS", created_by="u",
	)))
	interaction = run(s.record_interaction(DrugInteractionCreate(
		tenant_id="t", drug_a_id=drug_a.id, drug_b_id=drug_b.id,
		severity="major", mechanism="CYP2C9 inhibition",
		clinical_effect="Increased warfarin levels", management="Monitor INR",
		evidence_source="Lexicomp", created_by="pharmacist1",
	)))
	assert interaction.severity == "major"


def test_check_interactions_returns_pair():
	s = svc()
	d1 = make_drug(s)
	d2 = run(s.add_drug_to_formulary(DrugCreate(tenant_id="t", drug_name="W", generic_name="w", ndc_code="0001", drug_type="generic", drug_schedule="non_controlled", dosage_form="tablet", strength="1", unit="mg", manufacturer="M", created_by="u")))
	run(s.record_interaction(DrugInteractionCreate(tenant_id="t", drug_a_id=d1.id, drug_b_id=d2.id, severity="moderate", mechanism="x", clinical_effect="y", management="z", evidence_source="src", created_by="u")))
	hits = run(s.check_interactions("t", [d1.id, d2.id]))
	assert len(hits) == 1


def test_controlled_substance_waste_requires_witness():
	s = svc()
	try:
		run(s.log_controlled_substance(ControlledSubstanceLogCreate(
			tenant_id="t", drug_id="drug1", drug_schedule="schedule_ii",
			action="waste", quantity=2.0, unit="mL", performed_by="nurse1",
			witness_id=None, created_by="nurse1",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_controlled_substance_waste_with_witness():
	s = svc()
	log = run(s.log_controlled_substance(ControlledSubstanceLogCreate(
		tenant_id="t", drug_id="drug1", drug_schedule="schedule_ii",
		action="waste", quantity=2.0, unit="mL", performed_by="nurse1",
		witness_id="nurse2", waste_amount=2.0, created_by="nurse1",
	)))
	assert log.action == "waste" and log.witness_id == "nurse2"


def test_add_inventory_and_expiry_status():
	s = svc()
	item = run(s.add_inventory_item(InventoryItemCreate(
		tenant_id="t", drug_id="drug1", lot_number="L001",
		quantity_on_hand=100.0, unit="tablets",
		expiry_date=datetime.utcnow() + timedelta(days=200),
		location="Shelf A", created_by="pharmacist1",
	)))
	assert item.status == "in_stock"
	expiring = run(s.add_inventory_item(InventoryItemCreate(
		tenant_id="t", drug_id="drug2", lot_number="L002",
		quantity_on_hand=10.0, unit="tablets",
		expiry_date=datetime.utcnow() + timedelta(days=15),
		location="Shelf B", created_by="pharmacist1",
	)))
	assert expiring.status == "low_stock"


def test_prior_auth_workflow():
	s = svc()
	pa = run(s.request_prior_auth(PriorAuthCreate(
		tenant_id="t", patient_id="p1", drug_id="drug1", prescription_id="rx1",
		insurance_id="ins1", diagnosis_icd10="E11.9",
		requested_by="dr1", clinical_justification="T2DM first line", created_by="dr1",
	)))
	assert pa.status == "pending"
	approved = run(s.approve_prior_auth("t", pa.id, "ins_reviewer"))
	assert approved.status == "approved" and approved.expires_at is not None


def test_dashboard_summary():
	s = svc()
	make_drug(s)
	summary = run(s.dashboard_summary("t"))
	assert summary["formulary"]["total"] == 1
	assert "dispensing" in summary and "inventory" in summary
