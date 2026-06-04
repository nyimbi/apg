"""Service tests for Lease Management (lea)."""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal

import pytest

from capabilities.realestate.lea.service import LeaService
from capabilities.realestate.lea.models import (
	LeaseCreate, LeaseType, LeaseStatus,
	LeaseAbstractionCreate,
	RentEscalationCreate, EscalationType,
	LeaseOptionCreate, OptionType,
	RentReviewCreate, RentReviewType,
	Ifrs16ScheduleCreate, Ifrs16Category,
	LeaseAssignmentCreate,
)

loop = asyncio.get_event_loop()
T = "test-tenant"


def _svc():
	return LeaService()


def _lease(svc, **kwargs):
	defaults = dict(
		tenant_id=T, property_id="prop-1", unit_id="unit-1",
		tenant_entity_id="ten-1", lease_type=LeaseType.commercial,
		lease_ref="L-001", commencement_date=date(2025, 1, 1),
		expiry_date=date(2030, 12, 31), initial_rent=Decimal("100000"),
		created_by="u",
	)
	defaults.update(kwargs)
	return loop.run_until_complete(svc.create_lease(LeaseCreate(**defaults)))


# ── Lease CRUD ────────────────────────────────────────────────────────────────

def test_create_lease():
	svc = _svc()
	l = _lease(svc)
	assert l.id
	assert l.status == LeaseStatus.heads_of_terms
	assert l.current_rent == Decimal("100000")


def test_get_lease():
	svc = _svc()
	l = _lease(svc)
	fetched = loop.run_until_complete(svc.get_lease(l.id, T))
	assert fetched.lease_ref == "L-001"


def test_list_leases_by_property():
	svc = _svc()
	_lease(svc, property_id="p1", lease_ref="L-A")
	_lease(svc, property_id="p1", lease_ref="L-B")
	_lease(svc, property_id="p2", lease_ref="L-C")
	results = loop.run_until_complete(svc.list_leases(T, property_id="p1"))
	assert len(results) == 2


def test_expiry_date_before_commencement_raises():
	svc = _svc()
	with pytest.raises(Exception):
		_lease(svc, commencement_date=date(2025, 6, 1), expiry_date=date(2025, 1, 1))


# ── Abstraction ───────────────────────────────────────────────────────────────

def test_create_and_verify_abstraction():
	svc = _svc()
	l = _lease(svc)
	abstr = loop.run_until_complete(svc.create_abstraction(LeaseAbstractionCreate(
		tenant_id=T, lease_id=l.id, source_document_id="doc-1", abstracted_by="ai",
	)))
	assert abstr.status.value == "pending"
	verified = loop.run_until_complete(svc.verify_abstraction(abstr.id, T, "verifier"))
	assert verified.status.value == "verified"
	# Lease should now be marked verified
	lease = loop.run_until_complete(svc.get_lease(l.id, T))
	assert lease.abstraction_verified is True


# ── Activation ────────────────────────────────────────────────────────────────

def test_activate_lease_requires_verified_abstraction():
	svc = _svc()
	l = _lease(svc)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.activate_lease(l.id, T))


def test_activate_lease_after_verification():
	svc = _svc()
	l = _lease(svc)
	abstr = loop.run_until_complete(svc.create_abstraction(LeaseAbstractionCreate(
		tenant_id=T, lease_id=l.id, source_document_id="doc-1", abstracted_by="ai",
	)))
	loop.run_until_complete(svc.verify_abstraction(abstr.id, T, "v"))
	activated = loop.run_until_complete(svc.activate_lease(l.id, T))
	assert activated.status == LeaseStatus.active


# ── Escalation ────────────────────────────────────────────────────────────────

def test_rent_escalation_applies():
	svc = _svc()
	l = _lease(svc)
	esc = loop.run_until_complete(svc.create_escalation(RentEscalationCreate(
		tenant_id=T, lease_id=l.id,
		escalation_type=EscalationType.fixed_percentage,
		effective_date=date(2026, 1, 1),
		escalation_rate=Decimal("0.05"),
		created_by="u",
	)))
	assert esc.old_rent == Decimal("100000")
	applied = loop.run_until_complete(svc.apply_escalation(esc.id, T, "u"))
	assert applied.applied is True
	lease = loop.run_until_complete(svc.get_lease(l.id, T))
	assert Decimal(str(lease.current_rent)) == Decimal("105000.00")


# ── Options ───────────────────────────────────────────────────────────────────

def test_create_option():
	svc = _svc()
	l = _lease(svc)
	opt = loop.run_until_complete(svc.create_option(LeaseOptionCreate(
		tenant_id=T, lease_id=l.id,
		option_type=OptionType.break_option_tenant,
		exercise_from=date(2025, 1, 1),
		exercise_to=date(2035, 12, 31),
		effective_date=date(2026, 1, 1),
		notice_required_days=90,
		created_by="u",
	)))
	assert opt.status == "open"


def test_exercise_option_without_notice_raises():
	svc = _svc()
	l = _lease(svc)
	opt = loop.run_until_complete(svc.create_option(LeaseOptionCreate(
		tenant_id=T, lease_id=l.id,
		option_type=OptionType.renewal_option,
		exercise_from=date(2025, 1, 1),
		exercise_to=date(2035, 12, 31),
		effective_date=date(2026, 1, 1),
		created_by="u",
	)))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.exercise_option(opt.id, T, notice_served=False))


# ── IFRS 16 ───────────────────────────────────────────────────────────────────

def test_ifrs16_schedule():
	svc = _svc()
	l = _lease(svc)
	schedule = loop.run_until_complete(svc.generate_ifrs16_schedule(Ifrs16ScheduleCreate(
		tenant_id=T, lease_id=l.id,
		category=Ifrs16Category.operating_lease,
		commencement_date=date(2025, 1, 1),
		expiry_date=date(2028, 12, 31),
		annual_payment=Decimal("60000"),
		discount_rate=Decimal("0.06"),
	)))
	assert schedule.rou_asset > 0
	assert schedule.lease_liability > 0
	# Lease should be updated
	lease = loop.run_until_complete(svc.get_lease(l.id, T))
	assert lease.ifrs16_category == Ifrs16Category.operating_lease


def test_ifrs16_invalid_discount_rate_raises():
	svc = _svc()
	with pytest.raises(Exception):
		Ifrs16ScheduleCreate(
			tenant_id=T, lease_id="l1",
			category=Ifrs16Category.finance_lease,
			commencement_date=date(2025, 1, 1),
			expiry_date=date(2030, 1, 1),
			annual_payment=Decimal("50000"),
			discount_rate=Decimal("1.5"),
		)


# ── Expiry Pipeline ───────────────────────────────────────────────────────────

def test_expiry_pipeline():
	svc = _svc()
	abstr = None
	l = _lease(svc, expiry_date=date(2025, 3, 1))
	# Activate lease
	abstr = loop.run_until_complete(svc.create_abstraction(LeaseAbstractionCreate(
		tenant_id=T, lease_id=l.id, source_document_id="d", abstracted_by="ai",
	)))
	loop.run_until_complete(svc.verify_abstraction(abstr.id, T, "v"))
	loop.run_until_complete(svc.activate_lease(l.id, T))
	pipeline = loop.run_until_complete(svc.get_expiry_pipeline(T, months_ahead=24))
	assert any(e["lease_id"] == l.id for e in pipeline)


# ── Surrender ─────────────────────────────────────────────────────────────────

def test_surrender_inactive_lease_raises():
	svc = _svc()
	l = _lease(svc)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.surrender_lease(l.id, T, "u"))
