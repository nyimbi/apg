"""Service tests for Real Estate Accounting (acc)."""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal

import pytest

from capabilities.realestate.acc.service import AccService
from capabilities.realestate.acc.models import (
	AccountCreate, AccountType, LedgerType,
	JournalEntryCreate, JournalLine, JournalType,
	ServiceChargeCreate, ChargeType,
	CamReconciliationCreate,
	Ifrs16ScheduleCreate, Ifrs16Category,
	RevenueScheduleCreate, RevenueMethod,
	AccountingPeriodCreate,
	TenantStatementCreate,
)

loop = asyncio.get_event_loop()
T = "test-tenant"


def _svc():
	return AccService()


# ── Account ───────────────────────────────────────────────────────────────────

def test_create_and_get_account():
	svc = _svc()
	payload = AccountCreate(
		tenant_id=T, code="1000", name="Property Ledger",
		account_type=AccountType.asset, ledger_type=LedgerType.property_ledger,
		created_by="user1",
	)
	acc = loop.run_until_complete(svc.create_account(payload))
	assert acc.id
	assert acc.code == "1000"
	fetched = loop.run_until_complete(svc.get_account(acc.id, T))
	assert fetched.name == "Property Ledger"


def test_list_accounts_filtered_by_property():
	svc = _svc()
	for i in range(3):
		loop.run_until_complete(svc.create_account(AccountCreate(
			tenant_id=T, code=f"A{i}", name=f"Acc{i}",
			account_type=AccountType.revenue, ledger_type=LedgerType.rental_income,
			property_id="prop-1" if i < 2 else "prop-2", created_by="u",
		)))
	filtered = loop.run_until_complete(svc.list_accounts(T, property_id="prop-1"))
	assert len(filtered) == 2


def test_update_account():
	svc = _svc()
	from capabilities.realestate.acc.models import AccountUpdate
	acc = loop.run_until_complete(svc.create_account(AccountCreate(
		tenant_id=T, code="2000", name="Old Name",
		account_type=AccountType.expense, ledger_type=LedgerType.opex, created_by="u",
	)))
	updated = loop.run_until_complete(svc.update_account(acc.id, T, AccountUpdate(name="New Name")))
	assert updated.name == "New Name"


# ── Journal Entry ─────────────────────────────────────────────────────────────

def _make_balanced_lines():
	return [
		JournalLine(account_id="a1", account_code="1000", description="Dr", debit=Decimal("1000"), credit=Decimal("0")),
		JournalLine(account_id="a2", account_code="2000", description="Cr", debit=Decimal("0"), credit=Decimal("1000")),
	]


def test_create_journal_entry():
	svc = _svc()
	payload = JournalEntryCreate(
		tenant_id=T, journal_type=JournalType.manual, reference="JNL-001",
		period="2025-01", journal_date=date(2025, 1, 15),
		description="Test entry", lines=_make_balanced_lines(), created_by="u",
	)
	j = loop.run_until_complete(svc.create_journal_entry(payload))
	assert j.id
	assert j.total_debit == Decimal("1000")


def test_unbalanced_journal_raises():
	svc = _svc()
	lines = [JournalLine(account_id="a1", account_code="1000", description="Dr", debit=Decimal("500"), credit=Decimal("0"))]
	with pytest.raises(Exception):
		JournalEntryCreate(
			tenant_id=T, journal_type=JournalType.manual, reference="X",
			period="2025-01", journal_date=date(2025, 1, 1),
			description="unbalanced", lines=lines, created_by="u",
		)


def test_approve_and_post_journal():
	svc = _svc()
	payload = JournalEntryCreate(
		tenant_id=T, journal_type=JournalType.manual, reference="JNL-002",
		period="2025-02", journal_date=date(2025, 2, 1),
		description="Approve+Post", lines=_make_balanced_lines(), created_by="u",
	)
	j = loop.run_until_complete(svc.create_journal_entry(payload))
	approved = loop.run_until_complete(svc.approve_journal_entry(j.id, T, "approver1"))
	assert approved.status.value == "approved"
	posted = loop.run_until_complete(svc.post_journal_entry(j.id, T))
	assert posted.status.value == "posted"
	assert posted.posted_at is not None


def test_post_unapproved_journal_raises():
	svc = _svc()
	payload = JournalEntryCreate(
		tenant_id=T, journal_type=JournalType.manual, reference="JNL-003",
		period="2025-03", journal_date=date(2025, 3, 1),
		description="Not approved", lines=_make_balanced_lines(), created_by="u",
	)
	j = loop.run_until_complete(svc.create_journal_entry(payload))
	with pytest.raises(ValueError, match="approved"):
		loop.run_until_complete(svc.post_journal_entry(j.id, T))


# ── Service Charge ────────────────────────────────────────────────────────────

def test_raise_and_approve_service_charge():
	svc = _svc()
	payload = ServiceChargeCreate(
		tenant_id=T, property_id="prop-1", charge_type=ChargeType.base_rent,
		description="Jan Rent", amount=Decimal("50000"),
		period="2025-01", due_date=date(2025, 1, 31), created_by="u",
	)
	c = loop.run_until_complete(svc.raise_service_charge(payload))
	assert c.total_amount == Decimal("50000")
	approved = loop.run_until_complete(svc.approve_service_charge(c.id, T, "mgr"))
	assert approved.approved_by == "mgr"


def test_service_charge_vat_calculation():
	svc = _svc()
	payload = ServiceChargeCreate(
		tenant_id=T, property_id="prop-1", charge_type=ChargeType.service_charge,
		description="SC", amount=Decimal("10000"), vat_rate=Decimal("0.16"),
		period="2025-01", due_date=date(2025, 1, 31), created_by="u",
	)
	c = loop.run_until_complete(svc.raise_service_charge(payload))
	assert c.vat_amount == Decimal("1600")
	assert c.total_amount == Decimal("11600")


# ── CAM Reconciliation ────────────────────────────────────────────────────────

def test_cam_reconciliation_workflow():
	svc = _svc()
	payload = CamReconciliationCreate(
		tenant_id=T, property_id="prop-1", period_year=2024,
		estimated_costs=Decimal("500000"), actual_costs=Decimal("520000"),
		lease_ids=["lease-1", "lease-2"], created_by="u",
	)
	cam = loop.run_until_complete(svc.start_cam_reconciliation(payload))
	assert cam.variance == Decimal("20000")
	approved = loop.run_until_complete(svc.approve_cam_reconciliation(cam.id, T, "fm"))
	assert approved.status.value == "approved"
	settled = loop.run_until_complete(svc.settle_cam_reconciliation(cam.id, T))
	assert settled.status.value == "settled"


def test_cam_settle_before_approve_raises():
	svc = _svc()
	payload = CamReconciliationCreate(
		tenant_id=T, property_id="p1", period_year=2024,
		estimated_costs=Decimal("100000"), actual_costs=Decimal("110000"),
		lease_ids=["l1"], created_by="u",
	)
	cam = loop.run_until_complete(svc.start_cam_reconciliation(payload))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.settle_cam_reconciliation(cam.id, T))


# ── IFRS 16 ───────────────────────────────────────────────────────────────────

def test_ifrs16_schedule_generates_rou_and_liability():
	svc = _svc()
	payload = Ifrs16ScheduleCreate(
		tenant_id=T, lease_id="l1",
		category=Ifrs16Category.operating_lease,
		commencement_date=date(2025, 1, 1),
		expiry_date=date(2030, 12, 31),
		annual_payment=Decimal("120000"),
		discount_rate=Decimal("0.08"),
		created_by="u",
	)
	s = loop.run_until_complete(svc.generate_ifrs16_schedule(payload))
	assert s.rou_asset > 0
	assert s.lease_liability > 0
	assert len(s.schedule_lines) > 0


# ── Period Management ─────────────────────────────────────────────────────────

def test_open_and_close_period():
	svc = _svc()
	payload = AccountingPeriodCreate(tenant_id=T, period="2025-05", opened_by="u")
	period = loop.run_until_complete(svc.open_period(payload))
	assert period.is_open is True
	closed = loop.run_until_complete(svc.close_period(period.id, T, "user1", "user2"))
	assert closed.is_open is False
	assert closed.closed_by == "user1"
	assert closed.second_approver == "user2"


def test_close_period_same_approver_raises():
	svc = _svc()
	p = loop.run_until_complete(svc.open_period(AccountingPeriodCreate(tenant_id=T, period="2025-06", opened_by="u")))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.close_period(p.id, T, "same_user", "same_user"))


# ── Tenant Statement ──────────────────────────────────────────────────────────

def test_generate_tenant_statement():
	svc = _svc()
	payload = TenantStatementCreate(
		tenant_id=T, property_id="p1", lease_id="l1",
		statement_period="2025-01", opening_balance=Decimal("5000"), created_by="u",
	)
	stmt = loop.run_until_complete(svc.generate_tenant_statement(payload))
	assert stmt.id
	assert stmt.opening_balance == Decimal("5000")


# ── Tax ───────────────────────────────────────────────────────────────────────

def test_calculate_tax():
	svc = _svc()
	result = loop.run_until_complete(svc.calculate_tax(T, Decimal("100000"), "vat", Decimal("0.16")))
	assert result["tax_amount"] == 16000.0
	assert result["gross_amount"] == 116000.0


# ── Financial Summary ─────────────────────────────────────────────────────────

def test_financial_summary():
	svc = _svc()
	summary = loop.run_until_complete(svc.get_financial_summary(T))
	assert "tenant_id" in summary
	assert "total_service_charges" in summary
