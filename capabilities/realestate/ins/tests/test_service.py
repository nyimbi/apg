"""Service tests for Property Insurance (ins)."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from decimal import Decimal

import pytest

from capabilities.realestate.ins.service import InsService
from capabilities.realestate.ins.models import (
	InsurerCreate, InsurerGrade,
	PolicyCreate, PolicyUpdate, PolicyType, CoverageStatus, ValuationBasis,
	InsuredAssetCreate, AssetType,
	ClaimCreate, ClaimType, ClaimStatus,
	EndorsementCreate, EndorsementType,
	PremiumAllocationCreate,
	CoverageGapCreate,
)

loop = asyncio.get_event_loop()
T = "test-tenant"


def _svc():
	return InsService()


def _insurer(svc, grade=InsurerGrade.preferred):
	return loop.run_until_complete(svc.register_insurer(InsurerCreate(
		tenant_id=T, name="SafeGuard Insurance",
		grade=grade, email="sg@test.com", created_by="u",
	)))


def _policy(svc, insurer_id, **kwargs):
	defaults = dict(
		tenant_id=T, policy_number="POL-001",
		policy_type=PolicyType.property_all_risk,
		insurer_id=insurer_id, property_ids=["prop-1"],
		commencement_date=date(2025, 1, 1),
		expiry_date=date(2025, 12, 31),
		sum_insured=Decimal("100000000"),
		annual_premium=Decimal("500000"),
		valuation_basis=ValuationBasis.reinstatement_cost,
		perils_covered=["fire", "flood", "theft"],
		created_by="u",
	)
	defaults.update(kwargs)
	return loop.run_until_complete(svc.create_policy(PolicyCreate(**defaults)))


# ── Insurer ───────────────────────────────────────────────────────────────────

def test_register_insurer():
	svc = _svc()
	i = _insurer(svc)
	assert i.id
	fetched = loop.run_until_complete(svc.get_insurer(i.id, T))
	assert fetched.name == "SafeGuard Insurance"


def test_list_insurers_by_grade():
	svc = _svc()
	_insurer(svc, grade=InsurerGrade.preferred)
	_insurer(svc, grade=InsurerGrade.approved)
	preferred = loop.run_until_complete(svc.list_insurers(T, grade="preferred"))
	assert len(preferred) == 1


# ── Policy ────────────────────────────────────────────────────────────────────

def test_create_policy():
	svc = _svc()
	i = _insurer(svc)
	p = _policy(svc, i.id)
	assert p.id
	assert p.policy_number == "POL-001"


def test_bind_policy_without_asset_schedule_raises():
	svc = _svc()
	i = _insurer(svc)
	p = _policy(svc, i.id)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.bind_policy(p.id, T))


def test_bind_policy_with_asset_schedule():
	svc = _svc()
	i = _insurer(svc)
	p = _policy(svc, i.id)
	loop.run_until_complete(svc.add_asset_to_schedule(InsuredAssetCreate(
		tenant_id=T, policy_id=p.id, property_id="prop-1",
		asset_type=AssetType.building, description="Main Building",
		insured_value=Decimal("80000000"),
		valuation_basis=ValuationBasis.reinstatement_cost,
		created_by="u",
	)))
	bound = loop.run_until_complete(svc.bind_policy(p.id, T))
	assert bound.status == CoverageStatus.active


def test_suspended_insurer_bind_raises():
	svc = _svc()
	i = _insurer(svc, grade=InsurerGrade.suspended)
	p = _policy(svc, i.id)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.bind_policy(p.id, T))


# ── Claim ─────────────────────────────────────────────────────────────────────

def _active_policy(svc):
	i = _insurer(svc)
	p = _policy(svc, i.id)
	loop.run_until_complete(svc.add_asset_to_schedule(InsuredAssetCreate(
		tenant_id=T, policy_id=p.id, property_id="prop-1",
		asset_type=AssetType.building, description="Bldg",
		insured_value=Decimal("80000000"),
		valuation_basis=ValuationBasis.reinstatement_cost,
		created_by="u",
	)))
	return loop.run_until_complete(svc.bind_policy(p.id, T))


def test_lodge_claim():
	svc = _svc()
	p = _active_policy(svc)
	claim = loop.run_until_complete(svc.lodge_claim(ClaimCreate(
		tenant_id=T, policy_id=p.id, claim_type=ClaimType.partial_loss,
		peril="fire", incident_date=date(2025, 3, 15),
		description="Office fire", estimated_loss=Decimal("2000000"),
		property_id="prop-1", created_by="u",
	)))
	assert claim.ref.startswith("CLM-")
	assert claim.status == ClaimStatus.lodged


def test_claim_against_inactive_policy_raises():
	svc = _svc()
	i = _insurer(svc)
	p = _policy(svc, i.id)  # not bound = inactive
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.lodge_claim(ClaimCreate(
			tenant_id=T, policy_id=p.id, claim_type=ClaimType.partial_loss,
			peril="flood", incident_date=date(2025, 1, 1),
			description="Flood damage", estimated_loss=Decimal("500000"),
			property_id="prop-1", created_by="u",
		)))


def test_approve_and_settle_claim():
	svc = _svc()
	p = _active_policy(svc)
	claim = loop.run_until_complete(svc.lodge_claim(ClaimCreate(
		tenant_id=T, policy_id=p.id, claim_type=ClaimType.partial_loss,
		peril="fire", incident_date=date(2025, 2, 1),
		description="Fire", estimated_loss=Decimal("500000"),
		property_id="prop-1", created_by="u",
	)))
	approved = loop.run_until_complete(svc.approve_claim(claim.id, T, Decimal("450000"), senior_approved=False))
	assert approved.status == ClaimStatus.approved
	settled = loop.run_until_complete(svc.settle_claim(claim.id, T, Decimal("450000")))
	assert settled.status == ClaimStatus.settled


def test_settlement_exceeds_sum_insured_raises():
	svc = _svc()
	p = _active_policy(svc)
	claim = loop.run_until_complete(svc.lodge_claim(ClaimCreate(
		tenant_id=T, policy_id=p.id, claim_type=ClaimType.total_loss,
		peril="fire", incident_date=date(2025, 4, 1),
		description="Total loss", estimated_loss=Decimal("200000000"),
		property_id="prop-1", created_by="u",
	)))
	loop.run_until_complete(svc.approve_claim(claim.id, T, Decimal("200000000"), senior_approved=True))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.settle_claim(claim.id, T, Decimal("200000000")))


# ── Endorsement ───────────────────────────────────────────────────────────────

def test_issue_endorsement():
	svc = _svc()
	p = _active_policy(svc)
	end = loop.run_until_complete(svc.issue_endorsement(EndorsementCreate(
		tenant_id=T, policy_id=p.id,
		endorsement_type=EndorsementType.addition_of_property,
		effective_date=date(2025, 6, 1),
		description="Add warehouse",
		sum_insured_change=Decimal("10000000"),
		created_by="u",
	)))
	assert end.ref.startswith("END-")


# ── Coverage Gap ──────────────────────────────────────────────────────────────

def test_detect_gap_no_policy():
	svc = _svc()
	gaps = loop.run_until_complete(svc.detect_coverage_gaps(T, "uninsured-property"))
	assert len(gaps) == 1
	assert gaps[0].severity == "critical"


def test_no_gap_with_active_policy():
	svc = _svc()
	p = _active_policy(svc)
	gaps = loop.run_until_complete(svc.detect_coverage_gaps(T, "prop-1"))
	assert len(gaps) == 0


# ── Renewal Pipeline ──────────────────────────────────────────────────────────

def test_renewal_pipeline():
	svc = _svc()
	i = _insurer(svc)
	p = _policy(svc, i.id, expiry_date=date.today() + timedelta(days=30))
	loop.run_until_complete(svc.add_asset_to_schedule(InsuredAssetCreate(
		tenant_id=T, policy_id=p.id, property_id="prop-1",
		asset_type=AssetType.building, description="B",
		insured_value=Decimal("50000000"),
		valuation_basis=ValuationBasis.market_value,
		created_by="u",
	)))
	loop.run_until_complete(svc.bind_policy(p.id, T))
	pipeline = loop.run_until_complete(svc.get_renewal_pipeline(T, days_ahead=60))
	assert len(pipeline) == 1
