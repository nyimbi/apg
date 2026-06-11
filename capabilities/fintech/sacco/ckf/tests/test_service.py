"""Tests for CheckOffService — async, no mocks, real objects."""
from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from capabilities.fintech.sacco.ckf.models import (
	CheckOffStatus,
	DeductionFrequency,
)
from capabilities.fintech.sacco.ckf.service import CheckOffService

TENANT = "test_tenant"
LOAN_ID = "loan-001"
PRODUCT_ID = "sav-001"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def svc() -> CheckOffService:
	return CheckOffService()


@pytest.fixture
def loop():
	return asyncio.get_event_loop()


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


_employer_seq = 0

async def _make_employer(svc: CheckOffService, name: str = "Equity Bank") -> str:
	global _employer_seq
	_employer_seq += 1
	emp = await svc.register_employer(
		tenant_id=TENANT,
		name=name,
		registration_number=f"REG-{_employer_seq:06d}",
		payroll_contact="payroll@equitybank.co.ke",
		remittance_account="0123456789",
		check_off_agreement_date="2025-01-01",
		deduction_frequency=DeductionFrequency.MONTHLY,
	)
	return emp.id


async def _link_member(svc: CheckOffService, employer_id: str, member_id: str = "mem-001",
                        employee_no: str = "EMP001", salary: Decimal = Decimal("80000")) -> None:
	await svc.add_member_employer_link(
		tenant_id=TENANT,
		member_id=member_id,
		employer_id=employer_id,
		employee_number=employee_no,
		basic_salary=salary,
		effective_date="2025-01-01",
		member_name="Alice Wambui",
	)


def _stub_loan(svc: CheckOffService, member_id: str = "mem-001",
               principal: str = "5000", interest: str = "500") -> None:
	svc.register_loan_installment(TENANT, member_id, {
		"loan_id": LOAN_ID,
		"loan_number": "LN-0001",
		"installment_no": 3,
		"principal_due": principal,
		"interest_due": interest,
		"penalty": "0",
		"status": "due",
	})


def _stub_savings(svc: CheckOffService, member_id: str = "mem-001", amount: str = "2000") -> None:
	svc.register_savings_contribution(TENANT, member_id, {
		"product_id": PRODUCT_ID,
		"product_name": "Regular Savings",
		"monthly_amount": amount,
	})


# ── Employer CRUD ─────────────────────────────────────────────────────────────

async def test_register_employer(svc):
	emp = await svc.register_employer(
		tenant_id=TENANT,
		name="KCB Bank",
		registration_number="REG-KCB",
		payroll_contact="hr@kcb.co.ke",
		remittance_account="999888777",
		check_off_agreement_date="2025-06-01",
	)
	assert emp.id
	assert emp.name == "KCB Bank"
	assert emp.is_active


async def test_duplicate_registration_rejected(svc):
	# Register once with a known number
	await svc.register_employer(
		tenant_id=TENANT,
		name="Safaricom",
		registration_number="REG-DUPE-001",
		payroll_contact="x",
		remittance_account="x",
		check_off_agreement_date="2025-01-01",
	)
	# Second attempt with the SAME registration number must fail
	with pytest.raises(ValueError, match="already_registered"):
		await svc.register_employer(
			tenant_id=TENANT,
			name="Safaricom Clone",
			registration_number="REG-DUPE-001",
			payroll_contact="x",
			remittance_account="x",
			check_off_agreement_date="2025-01-01",
		)


async def test_update_employer(svc):
	eid = await _make_employer(svc)
	updated = await svc.update_employer(TENANT, eid, {"payroll_contact": "newpayroll@equitybank.co.ke"})
	assert updated.payroll_contact == "newpayroll@equitybank.co.ke"


async def test_deactivate_employer(svc):
	eid = await _make_employer(svc)
	result = await svc.deactivate_employer(TENANT, eid, "closed_operations")
	assert not result.is_active
	assert result.deactivation_reason == "closed_operations"


async def test_list_employers_active_only(svc):
	eid1 = await _make_employer(svc, "EmployerA")
	eid2 = await _make_employer(svc, "EmployerB")
	await svc.deactivate_employer(TENANT, eid2, "test")
	active = await svc.list_employers(TENANT, active_only=True)
	assert len(active) == 1
	assert active[0].id == eid1


async def test_unknown_employer_raises(svc):
	with pytest.raises(KeyError):
		await svc.get_employer(TENANT, "nonexistent")


# ── Member Links ──────────────────────────────────────────────────────────────

async def test_add_member_employer_link(svc):
	eid = await _make_employer(svc)
	await _link_member(svc, eid)
	deductions = await svc.get_member_deductions(TENANT, "mem-001")
	assert deductions.employer_id == eid
	assert deductions.employee_number == "EMP001"
	assert deductions.basic_salary == Decimal("80000.00")


async def test_add_link_supersedes_old(svc):
	eid1 = await _make_employer(svc, "Bank A")
	eid2 = await _make_employer(svc, "Bank B")
	await _link_member(svc, eid1, "mem-X")
	await _link_member(svc, eid2, "mem-X", "EMP-NEW")
	# Should now be linked to eid2
	ded = await svc.get_member_deductions(TENANT, "mem-X")
	assert ded.employer_id == eid2


async def test_remove_link(svc):
	eid = await _make_employer(svc)
	await _link_member(svc, eid, "mem-002")
	await svc.remove_member_employer_link(TENANT, "mem-002", eid, "2026-01-01", "resigned")
	with pytest.raises(KeyError):
		await svc.get_member_deductions(TENANT, "mem-002")


async def test_member_deductions_with_loan_and_savings(svc):
	eid = await _make_employer(svc)
	await _link_member(svc, eid)
	_stub_loan(svc, "mem-001", "5000", "500")
	_stub_savings(svc, "mem-001", "2000")
	ded = await svc.get_member_deductions(TENANT, "mem-001")
	assert ded.total_loan_deductions == Decimal("5500.00")
	assert ded.total_savings_deductions == Decimal("2000.00")
	assert ded.total_deductions == Decimal("7500.00")


# ── Schedule ──────────────────────────────────────────────────────────────────

async def test_generate_schedule_single_member(svc):
	eid = await _make_employer(svc)
	await _link_member(svc, eid)
	_stub_loan(svc)
	_stub_savings(svc)
	sched = await svc.generate_check_off_schedule(TENANT, eid, payroll_month=6, payroll_year=2026)
	assert sched.total_members == 1
	assert sched.grand_total == Decimal("7500.00")
	assert sched.status == CheckOffStatus.PENDING
	assert sched.period_label == "June 2026"


async def test_generate_schedule_multiple_members(svc):
	eid = await _make_employer(svc)
	await _link_member(svc, eid, "mem-A", "EMP-A", Decimal("50000"))
	await _link_member(svc, eid, "mem-B", "EMP-B", Decimal("60000"))
	_stub_loan(svc, "mem-A", "3000", "300")
	_stub_savings(svc, "mem-A", "1000")
	_stub_loan(svc, "mem-B", "4000", "400")
	_stub_savings(svc, "mem-B", "1500")
	sched = await svc.generate_check_off_schedule(TENANT, eid, 6, 2026)
	assert sched.total_members == 2
	assert sched.grand_total == Decimal("10200.00")


async def test_invalid_month_rejected(svc):
	eid = await _make_employer(svc)
	with pytest.raises(AssertionError):
		await svc.generate_check_off_schedule(TENANT, eid, payroll_month=13, payroll_year=2026)


# ── Upload ────────────────────────────────────────────────────────────────────

async def test_upload_check_off_file(svc):
	eid = await _make_employer(svc)
	await _link_member(svc, eid)
	_stub_loan(svc)
	_stub_savings(svc)
	await svc.generate_check_off_schedule(TENANT, eid, 6, 2026)
	result = await svc.upload_check_off_file(
		TENANT, eid, 6, 2026,
		[{"member_id": "mem-001", "amount_received": "7500", "loan_deductions": "5500", "savings_deductions": "2000"}],
	)
	assert result["rows_accepted"] == 1
	assert result["status"] == CheckOffStatus.UPLOADED.value


# ── Reconciliation ────────────────────────────────────────────────────────────

async def test_reconcile_full_payment(svc):
	eid = await _make_employer(svc)
	await _link_member(svc, eid)
	_stub_loan(svc)
	_stub_savings(svc)
	await svc.generate_check_off_schedule(TENANT, eid, 6, 2026)
	await svc.upload_check_off_file(
		TENANT, eid, 6, 2026,
		[{"member_id": "mem-001", "amount_received": "7500", "loan_deductions": "5500", "savings_deductions": "2000"}],
	)
	result = await svc.reconcile_check_off(TENANT, eid, 6, 2026)
	assert result.status == CheckOffStatus.RECONCILED
	assert result.total_variance == Decimal("0.00")
	assert not result.demand_notice_required


async def test_reconcile_short_payment(svc):
	eid = await _make_employer(svc)
	await _link_member(svc, eid)
	_stub_loan(svc)
	_stub_savings(svc)
	await svc.generate_check_off_schedule(TENANT, eid, 6, 2026)
	await svc.upload_check_off_file(
		TENANT, eid, 6, 2026,
		[{"member_id": "mem-001", "amount_received": "5000", "loan_deductions": "5000", "savings_deductions": "0"}],
	)
	result = await svc.reconcile_check_off(TENANT, eid, 6, 2026)
	assert result.status == CheckOffStatus.SHORT_PAID
	assert result.demand_notice_required
	assert result.total_variance == Decimal("-2500.00")


async def test_reconcile_over_payment(svc):
	eid = await _make_employer(svc)
	await _link_member(svc, eid)
	_stub_loan(svc)
	_stub_savings(svc)
	await svc.generate_check_off_schedule(TENANT, eid, 6, 2026)
	await svc.upload_check_off_file(
		TENANT, eid, 6, 2026,
		[{"member_id": "mem-001", "amount_received": "9000", "loan_deductions": "5500", "savings_deductions": "2000"}],
	)
	result = await svc.reconcile_check_off(TENANT, eid, 6, 2026)
	assert result.status == CheckOffStatus.OVER_PAID
	assert result.excess_to_savings == Decimal("1500.00")


async def test_reconcile_without_upload_raises(svc):
	eid = await _make_employer(svc)
	with pytest.raises(ValueError, match="no_upload_found"):
		await svc.reconcile_check_off(TENANT, eid, 6, 2026)


# ── GL Posting ────────────────────────────────────────────────────────────────

async def _full_workflow(svc):
	eid = await _make_employer(svc)
	await _link_member(svc, eid)
	_stub_loan(svc)
	_stub_savings(svc)
	await svc.generate_check_off_schedule(TENANT, eid, 6, 2026)
	await svc.upload_check_off_file(
		TENANT, eid, 6, 2026,
		[{"member_id": "mem-001", "amount_received": "7500", "loan_deductions": "5500", "savings_deductions": "2000"}],
	)
	await svc.reconcile_check_off(TENANT, eid, 6, 2026)
	return eid


async def test_post_receipts(svc):
	eid = await _full_workflow(svc)
	result = await svc.post_check_off_receipts(TENANT, eid, 6, 2026)
	assert result["status"] == "posted"
	assert result["gl_entries_created"] == 2  # loan + savings
	assert Decimal(result["total_posted"]) == Decimal("7500.00")


async def test_post_receipts_idempotent(svc):
	eid = await _full_workflow(svc)
	await svc.post_check_off_receipts(TENANT, eid, 6, 2026)
	result2 = await svc.post_check_off_receipts(TENANT, eid, 6, 2026)
	assert result2["status"] == "already_posted"


async def test_post_without_reconcile_raises(svc):
	eid = await _make_employer(svc)
	with pytest.raises(ValueError, match="reconcile_first"):
		await svc.post_check_off_receipts(TENANT, eid, 6, 2026)


# ── Status & Queries ──────────────────────────────────────────────────────────

async def test_get_check_off_status(svc):
	eid = await _full_workflow(svc)
	await svc.post_check_off_receipts(TENANT, eid, 6, 2026)
	status = await svc.get_check_off_status(TENANT, eid, 6, 2026)
	assert status["is_posted"]
	assert status["employer_id"] == eid


async def test_outstanding_remittances(svc):
	eid = await _make_employer(svc)
	await _link_member(svc, eid)
	await svc.generate_check_off_schedule(TENANT, eid, 6, 2026)
	outstanding = await svc.get_outstanding_remittances(TENANT)
	assert len(outstanding) == 1
	assert outstanding[0]["employer_id"] == eid


async def test_send_reminder(svc):
	eid = await _make_employer(svc)
	await _link_member(svc, eid)
	await svc.generate_check_off_schedule(TENANT, eid, 6, 2026)
	result = await svc.send_remittance_reminder(TENANT, eid, 6, 2026)
	assert result["reminders_sent"] == 1
	result2 = await svc.send_remittance_reminder(TENANT, eid, 6, 2026)
	assert result2["reminders_sent"] == 2


async def test_employer_statement(svc):
	eid = await _full_workflow(svc)
	await svc.post_check_off_receipts(TENANT, eid, 6, 2026)
	stmt = await svc.generate_employer_statement(TENANT, eid, 1, 12, 2026, 2026)
	assert len(stmt.lines) == 1
	assert stmt.lines[0].period_label == "June 2026"
	assert stmt.total_expected == Decimal("7500.00")


async def test_member_check_off_history(svc):
	eid = await _full_workflow(svc)
	await svc.post_check_off_receipts(TENANT, eid, 6, 2026)
	history = await svc.get_member_check_off_history(TENANT, "mem-001", months=12)
	assert history.months_covered == 1
	assert history.total_loan_deducted == Decimal("5500.00")
	assert history.total_savings_deducted == Decimal("2000.00")


async def test_flag_employer_default(svc):
	eid = await _make_employer(svc)
	result = await svc.flag_employer_default(TENANT, eid, 5, 2026)
	assert result["defaulted_period"] == "May 2026"
	rem_key = f"{TENANT}:{eid}:2026:5"
	rem = svc._remittances[rem_key]
	assert rem["defaulted"]
	assert rem["check_off_status"] == CheckOffStatus.DEFAULTED.value


async def test_get_metrics(svc):
	eid = await _full_workflow(svc)
	await svc.upload_check_off_file(
		TENANT, eid, 6, 2026,
		[{"member_id": "mem-001", "amount_received": "7500", "loan_deductions": "5500", "savings_deductions": "2000"}],
	)
	metrics = await svc.get_check_off_metrics(TENANT, 6, 2026)
	assert metrics.total_employers == 1
	assert metrics.active_employers == 1
	assert metrics.total_members_on_checkoff == 1


async def test_batch_process_all_employers(svc):
	for i, name in enumerate(["Corp A", "Corp B", "Corp C"]):
		eid = await _make_employer(svc, name)
		await _link_member(svc, eid, f"mem-{i}", f"EMP-{i:03d}")
		_stub_loan(svc, f"mem-{i}")
		_stub_savings(svc, f"mem-{i}")
	result = await svc.batch_process_all_employers(TENANT, 7, 2026)
	assert result["employers_processed"] == 3
	assert result["employers_failed"] == 0


async def test_health_check(svc):
	health = await svc.health_check()
	assert health["status"] == "healthy"
	assert health["capability"] == "fintech_sacco_ckf"
