"""Service tests for Property Contracts (con)."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from decimal import Decimal

import pytest

from capabilities.realestate.con.service import ConService
from capabilities.realestate.con.models import (
	ContractCreate, ContractUpdate, ContractType, ContractStatus, ContractParty, PartyRole,
	ContractorCreate, ContractorGrade,
	MilestoneCreate, MilestoneType,
	VariationOrderCreate, VariationType,
	DisputeCreate, DisputeType,
	RetentionCreate, RetentionMethod,
	ClauseCreate,
)

loop = asyncio.get_event_loop()
T = "test-tenant"


def _svc():
	return ConService()


def _parties():
	return [
		ContractParty(party_id="p1", party_name="Buyer Corp", role=PartyRole.buyer),
		ContractParty(party_id="p2", party_name="Seller Ltd", role=PartyRole.seller),
	]


def _contract(svc, **kwargs):
	defaults = dict(
		tenant_id=T, contract_ref="CON-001",
		contract_type=ContractType.sale_purchase,
		parties=_parties(), governing_law="Kenya",
		start_date=date(2025, 1, 1), end_date=date(2026, 12, 31),
		contract_value=Decimal("5000000"),
		description="Sale of Plot 123",
		created_by="u",
	)
	defaults.update(kwargs)
	return loop.run_until_complete(svc.create_contract(ContractCreate(**defaults)))


# ── Contract ──────────────────────────────────────────────────────────────────

def test_create_contract():
	svc = _svc()
	c = _contract(svc)
	assert c.id
	assert c.status == ContractStatus.draft


def test_get_contract():
	svc = _svc()
	c = _contract(svc)
	fetched = loop.run_until_complete(svc.get_contract(c.id, T))
	assert fetched.contract_ref == "CON-001"


def test_list_contracts_by_type():
	svc = _svc()
	_contract(svc, contract_ref="A", contract_type=ContractType.sale_purchase)
	_contract(svc, contract_ref="B", contract_type=ContractType.management_contract)
	sale = loop.run_until_complete(svc.list_contracts(T, contract_type="sale_purchase"))
	assert len(sale) == 1


def test_contract_requires_two_parties():
	svc = _svc()
	with pytest.raises(Exception):
		ContractCreate(
			tenant_id=T, contract_ref="X", contract_type=ContractType.service_agreement,
			parties=[ContractParty(party_id="p1", party_name="Solo", role=PartyRole.buyer)],
			governing_law="Kenya", start_date=date(2025, 1, 1),
			description="Single party", created_by="u",
		)


def test_execute_contract_missing_signatures_raises():
	svc = _svc()
	c = _contract(svc)
	loop.run_until_complete(svc.update_contract(c.id, T, ContractUpdate(legal_review_complete=True)))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.execute_contract(c.id, T))


def test_execute_contract_full_workflow():
	svc = _svc()
	c = _contract(svc)
	# Sign both parties
	loop.run_until_complete(svc.sign_contract_party(c.id, T, "p1", "SIG-001"))
	loop.run_until_complete(svc.sign_contract_party(c.id, T, "p2", "SIG-002"))
	loop.run_until_complete(svc.update_contract(c.id, T, ContractUpdate(legal_review_complete=True)))
	executed = loop.run_until_complete(svc.execute_contract(c.id, T))
	assert executed.status == ContractStatus.active


def test_terminate_contract_no_reason_raises():
	svc = _svc()
	c = _contract(svc)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.terminate_contract(c.id, T, "", True))


def test_expiry_pipeline():
	svc = _svc()
	c = _contract(svc, end_date=date.today() + timedelta(days=30))
	# Execute contract first
	loop.run_until_complete(svc.sign_contract_party(c.id, T, "p1", "S1"))
	loop.run_until_complete(svc.sign_contract_party(c.id, T, "p2", "S2"))
	loop.run_until_complete(svc.update_contract(c.id, T, ContractUpdate(legal_review_complete=True)))
	loop.run_until_complete(svc.execute_contract(c.id, T))
	pipeline = loop.run_until_complete(svc.get_expiry_pipeline(T, days_ahead=60))
	assert any(e["contract_id"] == c.id for e in pipeline)


# ── Contractor ────────────────────────────────────────────────────────────────

def test_register_and_grade_contractor():
	svc = _svc()
	c = loop.run_until_complete(svc.register_contractor(ContractorCreate(
		tenant_id=T, name="BuildRight Ltd", contractor_type="construction_contract",
		email="build@test.com", phone="+254700000001", created_by="u",
	)))
	assert c.grade == ContractorGrade.conditional
	graded = loop.run_until_complete(svc.grade_contractor(c.id, T, ContractorGrade.preferred, "mgr"))
	assert graded.grade == ContractorGrade.preferred


# ── Milestone ─────────────────────────────────────────────────────────────────

def test_create_and_complete_milestone():
	svc = _svc()
	c = _contract(svc)
	m = loop.run_until_complete(svc.create_milestone(MilestoneCreate(
		tenant_id=T, contract_id=c.id, milestone_type=MilestoneType.payment,
		title="First Payment", due_date=date(2025, 3, 31), amount=Decimal("1000000"),
		created_by="u",
	)))
	assert m.status == "pending"
	completed = loop.run_until_complete(svc.complete_milestone(m.id, T, ["receipt.pdf"]))
	assert completed.status == "completed"


# ── Variation ─────────────────────────────────────────────────────────────────

def test_raise_variation_against_active_contract():
	svc = _svc()
	c = _contract(svc)
	loop.run_until_complete(svc.sign_contract_party(c.id, T, "p1", "S1"))
	loop.run_until_complete(svc.sign_contract_party(c.id, T, "p2", "S2"))
	loop.run_until_complete(svc.update_contract(c.id, T, ContractUpdate(legal_review_complete=True)))
	loop.run_until_complete(svc.execute_contract(c.id, T))
	vo = loop.run_until_complete(svc.raise_variation(VariationOrderCreate(
		tenant_id=T, contract_id=c.id, variation_type=VariationType.scope_change,
		description="Additional floor", amount_change=Decimal("200000"), created_by="u",
	)))
	assert vo.ref.startswith("VO-")


def test_raise_variation_on_draft_contract_raises():
	svc = _svc()
	c = _contract(svc)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.raise_variation(VariationOrderCreate(
			tenant_id=T, contract_id=c.id, variation_type=VariationType.price_adjustment,
			description="Price change", amount_change=Decimal("50000"), created_by="u",
		)))


# ── Dispute ───────────────────────────────────────────────────────────────────

def test_raise_and_resolve_dispute():
	svc = _svc()
	c = _contract(svc)
	d = loop.run_until_complete(svc.raise_dispute(DisputeCreate(
		tenant_id=T, contract_id=c.id, dispute_type=DisputeType.payment_dispute,
		description="Late payment penalty", raised_by="p1", created_by="u",
	)))
	assert d.status == "open"
	resolved = loop.run_until_complete(svc.resolve_dispute(d.id, T, "Settled via negotiation"))
	assert resolved.status == "resolved"


# ── Clause Library ────────────────────────────────────────────────────────────

def test_clause_library_search():
	svc = _svc()
	loop.run_until_complete(svc.create_clause(ClauseCreate(
		tenant_id=T, clause_type="payment_terms", title="Net 30 Payment",
		content="Payment due within 30 days of invoice.", created_by="u",
	)))
	results = loop.run_until_complete(svc.search_clauses(T, clause_type="payment_terms"))
	assert len(results) == 1
	no_results = loop.run_until_complete(svc.search_clauses(T, clause_type="penalty"))
	assert len(no_results) == 0
