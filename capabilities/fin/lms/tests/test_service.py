"""LMS service tests — async, no mocks, real objects.

Run: uv run pytest -vxs capabilities/fin/lms/tests/

© 2025 Datacraft. All rights reserved.
"""
from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
	AmortisationMethod, ClosureReason, DemandNoticeType, Loan, LoanClassification,
	LoanStatus, MoratoriumType, PaymentMethod, PenaltyType, RestructureType,
)
from service import LoanManagementService


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_svc() -> LoanManagementService:
	return LoanManagementService()


async def _seed_loan(svc: LoanManagementService, **overrides) -> Loan:
	"""Create and persist a PENDING_DISBURSEMENT loan."""
	kwargs: dict = dict(
		tenant_id="t1",
		customer_id="cust-001",
		product_code="TERM-12",
		principal=Decimal("100000"),
		rate=Decimal("0.14"),
		tenor_months=12,
		method=AmortisationMethod.REDUCING_BALANCE,
	)
	kwargs.update(overrides)
	loan = Loan(**kwargs)
	await svc._loans.save(loan.model_dump())
	return loan


# ── Health check ──────────────────────────────────────────────────────────────

def test_health_check():
	svc = _make_svc()
	h = svc.health_check()
	assert h["status"] == "healthy"
	assert h["service"] == "fin_lms"


# ── Schedule generation ───────────────────────────────────────────────────────

async def test_schedule_reducing_balance():
	svc = _make_svc()
	schedule = await svc.generate_amortisation_schedule(
		loan_id="loan-1",
		principal=Decimal("120000"),
		rate=Decimal("0.12"),
		tenor_months=12,
		method=AmortisationMethod.REDUCING_BALANCE,
		first_payment_date=date(2025, 2, 1),
	)
	assert len(schedule) == 12
	# Last balance should be 0
	assert Decimal(str(schedule[-1]["balance"])) == Decimal("0")
	# Instalments monotonically cover principal
	total_principal = sum(Decimal(str(r["principal"])) for r in schedule)
	assert abs(total_principal - Decimal("120000")) <= Decimal("1")


async def test_schedule_flat_rate():
	svc = _make_svc()
	schedule = await svc.generate_amortisation_schedule(
		loan_id="loan-fr",
		principal=Decimal("60000"),
		rate=Decimal("0.15"),
		tenor_months=6,
		method=AmortisationMethod.FLAT_RATE,
		first_payment_date=date(2025, 2, 1),
	)
	assert len(schedule) == 6
	# All interest amounts equal for flat rate
	interests = [Decimal(str(r["interest"])) for r in schedule]
	assert all(i == interests[0] for i in interests)


async def test_schedule_french_annuity():
	svc = _make_svc()
	schedule = await svc.generate_amortisation_schedule(
		loan_id="loan-fa",
		principal=Decimal("100000"),
		rate=Decimal("0.12"),
		tenor_months=12,
		method=AmortisationMethod.FRENCH_ANNUITY,
		first_payment_date=date(2025, 2, 1),
	)
	assert len(schedule) == 12
	# All PMT totals within 1 unit of each other (rounding residue on last)
	totals = [Decimal(str(r["total"])) for r in schedule]
	assert max(totals) - min(totals[:-1]) <= Decimal("2")


async def test_schedule_bullet():
	svc = _make_svc()
	schedule = await svc.generate_amortisation_schedule(
		loan_id="loan-b",
		principal=Decimal("50000"),
		rate=Decimal("0.10"),
		tenor_months=6,
		method=AmortisationMethod.BULLET,
		first_payment_date=date(2025, 2, 1),
	)
	assert len(schedule) == 6
	# Principal only at last instalment
	for row in schedule[:-1]:
		assert Decimal(str(row["principal"])) == Decimal("0")
	assert Decimal(str(schedule[-1]["principal"])) == Decimal("50000")


# ── Disbursement ──────────────────────────────────────────────────────────────

async def test_disburse_loan():
	svc = _make_svc()
	loan = await _seed_loan(svc)
	result = await svc.disburse_loan(
		tenant_id="t1",
		loan_id=loan.id,
		disbursement_date=date(2025, 1, 15),
		account_id="ACC-001",
		amount=Decimal("100000"),
		disbursement_ref="DRF-001",
	)
	assert result["disbursed_amount"] == "100000"
	assert len(result["schedule"]) == 12
	assert result["gl_entry_id"]

	updated = await svc.get_loan("t1", loan.id)
	assert updated["status"] == LoanStatus.ACTIVE.value


async def test_disburse_wrong_status_raises():
	svc = _make_svc()
	loan = await _seed_loan(svc)
	await svc.disburse_loan("t1", loan.id, date(2025, 1, 1), "ACC", Decimal("100000"), "R1")
	with pytest.raises(ValueError, match="PENDING_DISBURSEMENT"):
		await svc.disburse_loan("t1", loan.id, date(2025, 1, 1), "ACC", Decimal("100000"), "R1")


# ── Repayment ─────────────────────────────────────────────────────────────────

async def _disbursed_loan(svc: LoanManagementService, tenor: int = 12) -> Loan:
	loan = await _seed_loan(svc, tenor_months=tenor)
	await svc.disburse_loan("t1", loan.id, date(2025, 1, 15), "ACC-1", Decimal("100000"), "DRF-1")
	return loan


async def test_repayment_reduces_balance():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	result = await svc.record_repayment(
		tenant_id="t1",
		loan_id=loan.id,
		amount=Decimal("9000"),
		payment_date=date(2025, 2, 15),
		payment_ref="PAY-001",
		payment_method=PaymentMethod.MOBILE_MONEY,
	)
	balance = Decimal(result["remaining_balance"])
	assert balance < Decimal("100000")


async def test_full_repayment_closes_loan():
	svc = _make_svc()
	loan = await _seed_loan(svc, principal=Decimal("10000"), tenor_months=1,
		rate=Decimal("0.12"), method=AmortisationMethod.BULLET)
	await svc.disburse_loan("t1", loan.id, date(2025, 1, 1), "ACC", Decimal("10000"), "D1")

	# Pay everything in one shot
	await svc.record_repayment(
		tenant_id="t1", loan_id=loan.id, amount=Decimal("10833.33"),
		payment_date=date(2025, 2, 1), payment_ref="P1",
		payment_method=PaymentMethod.BANK_TRANSFER,
	)
	updated = await svc.get_loan("t1", loan.id)
	assert updated["status"] == LoanStatus.CLOSED.value


async def test_waterfall_order():
	"""Penalties cleared before principal."""
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	# Manually inject a penalty
	raw = await svc._loans.get(loan.id, "t1")
	raw["total_penalties"] = "500.00"
	await svc._loans.save(raw)

	result = await svc.record_repayment(
		tenant_id="t1", loan_id=loan.id, amount=Decimal("600"),
		payment_date=date(2025, 2, 15), payment_ref="P2",
		payment_method=PaymentMethod.CASH,
	)
	assert Decimal(result["allocated"]["penalty"]) == Decimal("500.00")


# ── Arrears ───────────────────────────────────────────────────────────────────

async def test_arrears_no_payment():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	# first_payment is 2025-02-15; check on 2025-03-20 → missed one instalment
	arrears = await svc.calculate_arrears("t1", loan.id, date(2025, 3, 20))
	assert arrears.days_past_due > 0
	assert arrears.amount_in_arrears > Decimal("0")
	assert arrears.installments_missed >= 1


async def test_arrears_current_loan():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	# Check same day as disbursement — nothing due yet
	arrears = await svc.calculate_arrears("t1", loan.id, date(2025, 1, 15))
	assert arrears.days_past_due == 0
	assert arrears.npa_status is False


async def test_npa_status_after_90_days():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	arrears = await svc.calculate_arrears("t1", loan.id, date(2025, 7, 1))
	assert arrears.npa_status is True
	assert arrears.classification in (LoanClassification.SUBSTANDARD, LoanClassification.DOUBTFUL, LoanClassification.LOSS)


# ── Classification & provision ────────────────────────────────────────────────

async def test_classify_performing():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	cls = await svc.classify_loan("t1", loan.id)
	assert cls == LoanClassification.PERFORMING


async def test_required_provision_performing():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	req = await svc.calculate_required_provision("t1", loan.id)
	# 1% of 100000
	assert req == Decimal("1000.00")


async def test_post_provision_entry():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	result = await svc.post_provision_entry("t1", loan.id, Decimal("1000"), date(2025, 1, 31))
	assert result["gl_entry_id"]
	assert result["posted_provision"] == "1000"


# ── Restructuring ─────────────────────────────────────────────────────────────

async def test_extend_tenor():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	result = await svc.restructure_loan(
		tenant_id="t1", loan_id=loan.id,
		restructure_type=RestructureType.EXTEND_TENOR,
		new_terms={"additional_months": 6},
		effective_date=date(2025, 6, 1),
		approved_by="manager1",
	)
	assert result["new_tenor_months"] == 18
	assert result["gl_entry_id"]


async def test_reduce_rate():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	result = await svc.restructure_loan(
		tenant_id="t1", loan_id=loan.id,
		restructure_type=RestructureType.REDUCE_RATE,
		new_terms={"new_rate": "0.10"},
		effective_date=date(2025, 6, 1),
		approved_by="manager1",
	)
	assert result["new_rate"] == "0.10"


# ── Moratorium ────────────────────────────────────────────────────────────────

async def test_grant_moratorium():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	result = await svc.grant_moratorium(
		tenant_id="t1", loan_id=loan.id,
		from_date=date(2025, 3, 1), to_date=date(2025, 5, 31),
		moratorium_type=MoratoriumType.FULL,
		reason="COVID relief",
		approved_by="mgr",
		interest_accrues=False,
	)
	assert result["moratorium_id"]
	updated = await svc.get_loan("t1", loan.id)
	assert updated["status"] == LoanStatus.MORATORIUM.value


# ── Write-off / recovery ──────────────────────────────────────────────────────

async def test_write_off_and_recovery():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	# Write off
	wo = await svc.write_off_loan(
		tenant_id="t1", loan_id=loan.id,
		write_off_date=date(2026, 1, 1),
		reason="360+ DPD, no recovery prospects",
		approved_by="cro",
		write_off_amount=Decimal("95000"),
	)
	assert wo["status"] == LoanStatus.WRITTEN_OFF.value

	# Recovery
	rec = await svc.record_recovery(
		tenant_id="t1", loan_id=loan.id,
		amount=Decimal("95000"),
		recovery_date=date(2026, 6, 1),
		method="legal",
	)
	updated = await svc.get_loan("t1", loan.id)
	assert updated["status"] == LoanStatus.RECOVERED.value


# ── Portfolio quality ─────────────────────────────────────────────────────────

async def test_portfolio_quality():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	pq = await svc.get_portfolio_quality("t1", date(2025, 6, 1))
	assert pq.total_loans >= 1
	assert pq.total_portfolio >= Decimal("100000")


# ── Early settlement ──────────────────────────────────────────────────────────

async def test_early_settlement():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	result = await svc.get_early_settlement_amount("t1", loan.id, date(2025, 6, 1))
	assert Decimal(result["settlement_amount"]) > Decimal("0")
	assert Decimal(result["rebate"]) >= Decimal("0")


# ── Collections / notices ─────────────────────────────────────────────────────

async def test_demand_notice():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	result = await svc.send_demand_notice("t1", loan.id, DemandNoticeType.REMINDER)
	assert result["notice_type"] == "reminder"


async def test_refer_to_collections():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	result = await svc.refer_to_collections("t1", loan.id, "collector1", "90+ DPD")
	assert result["referred_by"] == "collector1"
	updated = await svc.get_loan("t1", loan.id)
	assert updated["referred_to_collections"] is True


# ── Reprice ───────────────────────────────────────────────────────────────────

async def test_reprice_loan():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	result = await svc.reprice_loan("t1", loan.id, Decimal("0.16"), date(2025, 6, 1), "mgr")
	assert result["new_rate"] == "0.16"
	assert result["old_rate"] == "0.14"


# ── Batch arrears ─────────────────────────────────────────────────────────────

async def test_batch_arrears_idempotent():
	svc = _make_svc()
	await _disbursed_loan(svc)
	r1 = await svc.batch_calculate_arrears("t1", date(2025, 6, 1))
	r2 = await svc.batch_calculate_arrears("t1", date(2025, 6, 1))
	assert r1["processed"] == r2["processed"]
	assert r1["errors"] == 0


# ── Close loan ────────────────────────────────────────────────────────────────

async def test_close_loan():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	result = await svc.close_loan("t1", loan.id, date(2025, 12, 31), ClosureReason.FULLY_PAID)
	assert result["status"] == "closed"


# ── Statement ─────────────────────────────────────────────────────────────────

async def test_loan_statement():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	await svc.record_repayment(
		tenant_id="t1", loan_id=loan.id, amount=Decimal("9000"),
		payment_date=date(2025, 2, 15), payment_ref="P1",
		payment_method=PaymentMethod.BANK_TRANSFER,
	)
	lines = await svc.get_loan_statement("t1", loan.id, date(2025, 1, 1), date(2025, 12, 31))
	assert len(lines) >= 1


# ── Guard validations ─────────────────────────────────────────────────────────

async def test_missing_tenant_raises():
	svc = _make_svc()
	with pytest.raises(ValueError, match="tenant_id"):
		await svc.get_loan("", "any")


async def test_missing_loan_raises():
	svc = _make_svc()
	with pytest.raises(KeyError):
		await svc.get_loan("t1", "nonexistent")


# ── Provision report ──────────────────────────────────────────────────────────

async def test_provision_report():
	svc = _make_svc()
	loan = await _disbursed_loan(svc)
	await svc.post_provision_entry("t1", loan.id, Decimal("1000"), date(2025, 1, 31))
	report = await svc.get_provision_report("t1", date(2025, 1, 31))
	assert Decimal(report["total_posted"]) == Decimal("1000.00")
