"""Service tests for Rental Operations (ren)."""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal

import pytest

from capabilities.realestate.ren.service import RenService
from capabilities.realestate.ren.models import (
	TenancyCreate, TenancyType, TenancyStatus, TenancyUpdate,
	RentPaymentCreate, PaymentMethod,
	DepositCreate, DepositType, DepositDeductionCreate,
	NoticeCreate, NoticeType,
	TenancyRenewalCreate,
	ReferencingCreate,
	ArrearsRecordCreate,
)

loop = asyncio.get_event_loop()
T = "test-tenant"


def _svc():
	return RenService()


def _tenancy(svc, **kwargs):
	defaults = dict(
		tenant_id=T, unit_id="unit-1", property_id="prop-1",
		tenant_entity_id="ten-1", tenancy_type=TenancyType.commercial,
		start_date=date(2025, 1, 1), end_date=date(2025, 12, 31),
		rent_amount=Decimal("50000"), rent_frequency="monthly",
		created_by="u",
	)
	defaults.update(kwargs)
	return loop.run_until_complete(svc.create_tenancy(TenancyCreate(**defaults)))


# ── Tenancy ───────────────────────────────────────────────────────────────────

def test_create_tenancy():
	svc = _svc()
	t = _tenancy(svc)
	assert t.id
	assert t.status == TenancyStatus.application


def test_get_tenancy():
	svc = _svc()
	t = _tenancy(svc)
	fetched = loop.run_until_complete(svc.get_tenancy(t.id, T))
	assert fetched.unit_id == "unit-1"


def test_list_tenancies_by_status():
	svc = _svc()
	t1 = _tenancy(svc, unit_id="u1")
	t2 = _tenancy(svc, unit_id="u2")
	all_t = loop.run_until_complete(svc.list_tenancies(T))
	assert len(all_t) == 2


def test_activate_tenancy_requires_deposit_and_referencing():
	svc = _svc()
	t = _tenancy(svc)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.activate_tenancy(t.id, T))


# ── Deposit + Activation ──────────────────────────────────────────────────────

def _register_deposit(svc, tenancy_id):
	return loop.run_until_complete(svc.register_deposit(DepositCreate(
		tenant_id=T, tenancy_id=tenancy_id,
		deposit_type=DepositType.cash_deposit, amount=Decimal("100000"),
		created_by="u",
	)))


def _complete_referencing(svc, tenancy_id):
	ref = loop.run_until_complete(svc.run_referencing(ReferencingCreate(
		tenant_id=T, tenancy_id=tenancy_id,
		referencing_types=["credit_check"], applicant_id="app-1", created_by="u",
	)))
	return loop.run_until_complete(svc.complete_referencing(ref.id, T, True, {"credit": "pass"}))


def test_full_activation_workflow():
	svc = _svc()
	t = _tenancy(svc)
	_register_deposit(svc, t.id)
	_complete_referencing(svc, t.id)
	activated = loop.run_until_complete(svc.activate_tenancy(t.id, T))
	assert activated.status == TenancyStatus.active


# ── Rent Collection ───────────────────────────────────────────────────────────

def test_record_full_rent_payment():
	svc = _svc()
	t = _tenancy(svc)
	payment = loop.run_until_complete(svc.record_rent_payment(RentPaymentCreate(
		tenant_id=T, tenancy_id=t.id, amount=Decimal("50000"),
		payment_date=date(2025, 2, 1), payment_method=PaymentMethod.bank_transfer,
		period="2025-02", created_by="u",
	)))
	assert payment.is_short_payment is False
	assert payment.shortfall == Decimal("0")


def test_record_short_payment_creates_arrears():
	svc = _svc()
	t = _tenancy(svc)
	payment = loop.run_until_complete(svc.record_rent_payment(RentPaymentCreate(
		tenant_id=T, tenancy_id=t.id, amount=Decimal("30000"),
		payment_date=date(2025, 2, 1), payment_method=PaymentMethod.mpesa,
		period="2025-02", created_by="u",
	)))
	assert payment.is_short_payment is True
	assert payment.shortfall == Decimal("20000")


# ── Deposit ───────────────────────────────────────────────────────────────────

def test_deposit_registered_updates_tenancy():
	svc = _svc()
	t = _tenancy(svc)
	dep = _register_deposit(svc, t.id)
	assert dep.status.value == "registered"
	tenancy = loop.run_until_complete(svc.get_tenancy(t.id, T))
	assert tenancy.deposit_registered is True


def test_deposit_deduction_requires_evidence():
	svc = _svc()
	t = _tenancy(svc)
	dep = _register_deposit(svc, t.id)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.deduct_from_deposit(DepositDeductionCreate(
			tenant_id=T, deposit_id=dep.id, reason="Damage",
			amount=Decimal("5000"), evidence_document_ids=[], created_by="u",
		)))


def test_deposit_deduction_with_evidence():
	svc = _svc()
	t = _tenancy(svc)
	dep = _register_deposit(svc, t.id)
	deduction = loop.run_until_complete(svc.deduct_from_deposit(DepositDeductionCreate(
		tenant_id=T, deposit_id=dep.id, reason="Damage",
		amount=Decimal("5000"), evidence_document_ids=["photo-1.jpg"], created_by="u",
	)))
	assert deduction.id
	# Check deposit total updated
	updated_dep = loop.run_until_complete(svc.get_deposit(dep.id, T))
	assert Decimal(str(updated_dep.total_deducted)) == Decimal("5000")


# ── Notice ────────────────────────────────────────────────────────────────────

def test_serve_notice_updates_tenancy_status():
	svc = _svc()
	t = _tenancy(svc)
	notice = loop.run_until_complete(svc.serve_notice(NoticeCreate(
		tenant_id=T, tenancy_id=t.id, notice_type=NoticeType.notice_to_quit,
		served_date=date(2025, 10, 1), effective_date=date(2025, 11, 1),
		served_by="landlord", created_by="u",
	)))
	assert notice.id
	tenancy = loop.run_until_complete(svc.get_tenancy(t.id, T))
	assert tenancy.status == TenancyStatus.notice_served


# ── Arrears ───────────────────────────────────────────────────────────────────

def test_arrears_classification():
	svc = _svc()
	from capabilities.realestate.ren.models import ArrearsStatus
	rec = loop.run_until_complete(svc.record_arrears(ArrearsRecordCreate(
		tenant_id=T, tenancy_id="t1", amount_overdue=Decimal("10000"),
		days_overdue=45, created_by="u",
	)))
	assert rec.status == ArrearsStatus.days_31_60


def test_legal_escalation_below_90_days_raises():
	svc = _svc()
	rec = loop.run_until_complete(svc.record_arrears(ArrearsRecordCreate(
		tenant_id=T, tenancy_id="t1", amount_overdue=Decimal("5000"),
		days_overdue=30, created_by="u",
	)))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.escalate_arrears_to_legal(rec.id, T))


# ── Renewal ───────────────────────────────────────────────────────────────────

def test_initiate_and_accept_renewal():
	svc = _svc()
	t = _tenancy(svc)
	renewal = loop.run_until_complete(svc.initiate_renewal(TenancyRenewalCreate(
		tenant_id=T, tenancy_id=t.id, renewal_type="fixed_term_renewal",
		new_start_date=date(2026, 1, 1), new_end_date=date(2026, 12, 31),
		new_rent=Decimal("55000"), created_by="u",
	)))
	assert renewal.status == "pending"
	accepted = loop.run_until_complete(svc.accept_renewal(renewal.id, T))
	assert accepted.status == "accepted"


# ── Rent Roll ─────────────────────────────────────────────────────────────────

def test_rent_roll_active_tenancies_only():
	svc = _svc()
	t1 = _tenancy(svc, unit_id="u1")
	t2 = _tenancy(svc, unit_id="u2")
	# Only one activated
	_register_deposit(svc, t1.id)
	_complete_referencing(svc, t1.id)
	loop.run_until_complete(svc.activate_tenancy(t1.id, T))
	roll = loop.run_until_complete(svc.generate_rent_roll(T))
	assert len(roll) == 1
	assert roll[0]["tenancy_id"] == t1.id
