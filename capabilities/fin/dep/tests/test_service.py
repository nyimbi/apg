"""Service-layer tests for fin.dep."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from capabilities.fin.dep.models import (
	BatchAccrualResult, CompoundingFrequency, FeeConfig, InterestCalculationType,
	InterestConfig, MaturityInstruction, ProductStatus, ProductTerms, ProductType,
)
from capabilities.fin.dep.service import DepositProductsService


# ── Product lifecycle ────────────────────────────────────────────────────────

def test_create_product(svc, tenant, savings_product):
	assert savings_product.code == "SAV001"
	assert savings_product.status == ProductStatus.ACTIVE
	assert savings_product.interest_config.rate == Decimal("5.5")


def test_create_duplicate_product_raises(svc, tenant, savings_product):
	from capabilities.fin.dep.models import InterestConfig, FeeConfig, ProductTerms, ProductType
	with pytest.raises(ValueError, match="already exists"):
		svc.create_product(
			tenant_id=tenant, code="SAV001", name="Dup",
			product_type=ProductType.SAVINGS, currency="KES",
			interest_config=InterestConfig(rate=Decimal("3")),
			fee_config=FeeConfig(), terms=ProductTerms(),
		)


def test_get_product(svc, tenant, savings_product):
	p = svc.get_product(tenant, "SAV001")
	assert p.name == "Classic Savings"


def test_get_missing_product_raises(svc, tenant):
	with pytest.raises(KeyError):
		svc.get_product(tenant, "NONE")


def test_list_products_all(svc, tenant, savings_product, term_product):
	products = svc.list_products(tenant)
	assert len(products) == 2


def test_list_products_by_type(svc, tenant, savings_product, term_product):
	savings = svc.list_products(tenant, product_type=ProductType.SAVINGS)
	assert all(p.product_type == ProductType.SAVINGS for p in savings)
	assert len(savings) == 1


def test_deactivate_product(svc, tenant, savings_product):
	p = svc.deactivate_product(tenant, "SAV001")
	assert p.status == ProductStatus.INACTIVE
	# Inactive excluded from active_only listing
	active = svc.list_products(tenant, active_only=True)
	assert not any(p2.code == "SAV001" for p2 in active)


def test_update_product(svc, tenant, savings_product):
	updated = svc.update_product(tenant, "SAV001", {"name": "Premium Savings"})
	assert updated.name == "Premium Savings"


def test_invalid_tenant_raises(svc):
	with pytest.raises(ValueError, match="tenant_id is required"):
		svc.list_products("")


# ── Rate management ──────────────────────────────────────────────────────────

def test_update_product_rate(svc, tenant, savings_product):
	entry = svc.update_product_rate(
		tenant, "SAV001", Decimal("6.5"), date(2025, 7, 1), "rate_review"
	)
	assert entry.new_rate == Decimal("6.5")
	assert entry.old_rate == Decimal("5.5")
	product = svc.get_product(tenant, "SAV001")
	assert product.interest_config.rate == Decimal("6.5")


def test_rate_schedule_grows(svc, tenant, savings_product):
	svc.update_product_rate(tenant, "SAV001", Decimal("6.0"), date(2025, 6, 1), "q2")
	svc.update_product_rate(tenant, "SAV001", Decimal("6.5"), date(2025, 7, 1), "q3")
	schedule = svc.get_rate_schedule(tenant, "SAV001")
	# Created + 2 updates
	assert len(schedule) == 3


# ── Interest calculation ─────────────────────────────────────────────────────

def test_calculate_interest_simple(svc, tenant, term_product, term_account):
	result = svc.calculate_interest(
		tenant, "ACC-TD-001",
		date(2025, 1, 1), date(2025, 4, 1),
		Decimal("100000"), "TD001",
	)
	# 90 days at 9% simple
	expected_gross = Decimal("100000") * Decimal("9") / Decimal("100") * Decimal("90") / Decimal("365")
	assert abs(result.gross_interest - expected_gross) < Decimal("0.01")
	assert result.accrual_days == 90
	assert result.rate_applied == Decimal("9.0")


def test_calculate_interest_wht(svc, tenant, savings_product, savings_account):
	result = svc.calculate_interest(
		tenant, "ACC-SAV-001",
		date(2025, 1, 1), date(2025, 1, 31),
		Decimal("50000"), "SAV001",
	)
	assert result.withholding_tax > Decimal("0")
	assert result.net_interest == result.gross_interest - result.withholding_tax


def test_calculate_interest_tiered(svc, tenant, tiered_product):
	svc.register_account(tenant, "ACC-TIER-001", "TIER001", Decimal("150000"))
	result = svc.calculate_interest(
		tenant, "ACC-TIER-001",
		date(2025, 1, 1), date(2025, 12, 31),
		Decimal("150000"), "TIER001",
	)
	# Should use 5% tier for 150k balance
	assert result.rate_applied == Decimal("5.0")
	assert result.withholding_tax == Decimal("0")  # no WHT on this product


def test_calculate_interest_compound(svc, tenant):
	svc.create_product(
		tenant_id=tenant, code="COMP001", name="Compound Savings",
		product_type=ProductType.SAVINGS, currency="USD",
		interest_config=InterestConfig(
			rate=Decimal("10"),
			calculation=InterestCalculationType.COMPOUND,
			compounding=CompoundingFrequency.MONTHLY,
		),
		fee_config=FeeConfig(), terms=ProductTerms(),
	)
	svc.register_account(tenant, "ACC-COMP-001", "COMP001", Decimal("10000"))
	result = svc.calculate_interest(
		tenant, "ACC-COMP-001",
		date(2025, 1, 1), date(2026, 1, 1),
		Decimal("10000"), "COMP001",
	)
	# Compound over 365 days at 10% monthly should exceed simple interest
	simple_gross = Decimal("10000") * Decimal("10") / Decimal("100")
	assert result.gross_interest > simple_gross


def test_same_day_accrual_treated_as_one_day(svc, tenant, savings_product, savings_account):
	result = svc.calculate_interest(
		tenant, "ACC-SAV-001",
		date(2025, 3, 1), date(2025, 3, 1),
		Decimal("10000"), "SAV001",
	)
	assert result.accrual_days == 1


# ── Interest posting ─────────────────────────────────────────────────────────

def test_apply_interest_credits_account(svc, tenant, savings_product, savings_account):
	before = svc._accounts[(tenant, "ACC-SAV-001")]["balance"]
	posting = svc.apply_interest(
		tenant, "ACC-SAV-001", Decimal("500"),
		date(2025, 2, 1), "POST-001",
	)
	after = svc._accounts[(tenant, "ACC-SAV-001")]["balance"]
	# Net interest credited (gross - WHT)
	net = Decimal(posting["net_interest"])
	assert after == before + net


def test_apply_interest_returns_gl_refs(svc, tenant, savings_product, savings_account):
	posting = svc.apply_interest(
		tenant, "ACC-SAV-001", Decimal("200"),
		date(2025, 2, 28), "POST-002",
	)
	assert "gl_ref" in posting
	assert posting["gl_debit"] == "4001-INT-INCOME"


# ── Fee management ───────────────────────────────────────────────────────────

def test_maintenance_fee_applied(svc, tenant, savings_product, savings_account):
	before = svc._accounts[(tenant, "ACC-SAV-001")]["balance"]
	result = svc.apply_maintenance_fee(tenant, "ACC-SAV-001", date(2025, 2, 28))
	after = svc._accounts[(tenant, "ACC-SAV-001")]["balance"]
	assert result["fee_amount"] == "200"
	assert after == before - Decimal("200")


def test_below_minimum_fee(svc, tenant, savings_product):
	svc.register_account(tenant, "ACC-LOW-001", "SAV001", Decimal("500"))  # below 1000 minimum
	result = svc.apply_maintenance_fee(tenant, "ACC-LOW-001", date(2025, 2, 28))
	assert result["reason"] == "below_minimum_balance"
	assert result["fee_amount"] == "50"


def test_no_fee_when_zero_config(svc, tenant):
	svc.create_product(
		tenant_id=tenant, code="NOFEE", name="No Fee",
		product_type=ProductType.CURRENT, currency="KES",
		interest_config=InterestConfig(rate=Decimal("0")),
		fee_config=FeeConfig(), terms=ProductTerms(),
	)
	svc.register_account(tenant, "ACC-NOFEE", "NOFEE", Decimal("1000"))
	result = svc.apply_maintenance_fee(tenant, "ACC-NOFEE", date(2025, 2, 28))
	assert result["posted"] is False


def test_check_minimum_balance_meets(svc, tenant, savings_product, savings_account):
	check = svc.check_minimum_balance(tenant, "ACC-SAV-001")
	assert check.meets_minimum is True
	assert check.shortfall == Decimal("0")
	assert check.fee_applicable is False


def test_check_minimum_balance_fails(svc, tenant, savings_product):
	svc.register_account(tenant, "ACC-MIN-FAIL", "SAV001", Decimal("800"))
	check = svc.check_minimum_balance(tenant, "ACC-MIN-FAIL")
	assert check.meets_minimum is False
	assert check.shortfall == Decimal("200")
	assert check.fee_applicable is True


# ── Term deposit maturity ────────────────────────────────────────────────────

def test_process_maturity_payout(svc, tenant, term_product, term_account):
	record = svc.process_term_deposit_maturity(
		tenant, "ACC-TD-001", MaturityInstruction.PAYOUT
	)
	assert record.instruction == MaturityInstruction.PAYOUT
	assert record.principal == Decimal("100000")
	assert record.interest_earned > Decimal("0")
	# Account balance should be 0 after payout
	assert svc._accounts[(tenant, "ACC-TD-001")]["balance"] == Decimal("0")


def test_process_maturity_rollover(svc, tenant, term_product, term_account):
	record = svc.process_term_deposit_maturity(
		tenant, "ACC-TD-001", MaturityInstruction.ROLLOVER
	)
	assert record.rollover_ref != ""
	# Balance should be principal + net interest
	new_balance = svc._accounts[(tenant, "ACC-TD-001")]["balance"]
	assert new_balance > Decimal("100000")


def test_maturity_requires_term_deposit(svc, tenant, savings_product, savings_account):
	with pytest.raises(AssertionError):
		svc.process_term_deposit_maturity(tenant, "ACC-SAV-001", MaturityInstruction.PAYOUT)


def test_calculate_break_penalty(svc, tenant, term_product, term_account):
	penalty = svc.calculate_break_penalty(tenant, "ACC-TD-001", date(2025, 2, 1))
	# 50% of gross interest for 31 days
	assert penalty > Decimal("0")


# ── Batch accrual ────────────────────────────────────────────────────────────

def test_batch_accrue_interest(svc, tenant, savings_product, savings_account):
	result = svc.batch_accrue_interest(tenant, date(2025, 3, 15))
	assert isinstance(result, BatchAccrualResult)
	assert result.accounts_processed >= 1
	assert result.total_accrued > Decimal("0")
	assert result.entries_posted >= 1


def test_batch_accrual_idempotent(svc, tenant, savings_product, savings_account):
	r1 = svc.batch_accrue_interest(tenant, date(2025, 3, 15))
	r2 = svc.batch_accrue_interest(tenant, date(2025, 3, 15))
	assert r2.idempotent_hit is True
	assert r1.total_accrued == r2.total_accrued
	assert r1.entries_posted == r2.entries_posted


def test_batch_accrual_different_dates(svc, tenant, savings_product, savings_account):
	r1 = svc.batch_accrue_interest(tenant, date(2025, 3, 15))
	r2 = svc.batch_accrue_interest(tenant, date(2025, 3, 16))
	assert r2.idempotent_hit is False
	assert r1.accrual_date != r2.accrual_date


def test_batch_accrual_skips_inactive_products(svc, tenant, savings_product, savings_account):
	svc.deactivate_product(tenant, "SAV001")
	result = svc.batch_accrue_interest(tenant, date(2025, 4, 1))
	assert result.accounts_processed == 0


# ── Accrued interest ─────────────────────────────────────────────────────────

def test_get_accrued_interest_sums_unposted(svc, tenant, savings_product, savings_account):
	svc.batch_accrue_interest(tenant, date(2025, 3, 1))
	svc.batch_accrue_interest(tenant, date(2025, 3, 2))
	accrued = svc.get_accrued_interest(tenant, "ACC-SAV-001", date(2025, 3, 2))
	assert accrued > Decimal("0")


# ── Simulation ───────────────────────────────────────────────────────────────

def test_simulate_maturity(svc, tenant, term_product):
	result = svc.simulate_maturity(tenant, "TD001", Decimal("200000"), 90)
	assert result.principal == Decimal("200000")
	assert result.tenor_days == 90
	assert result.gross_interest > Decimal("0")
	assert result.maturity_amount == result.principal + result.net_interest
	assert result.effective_rate > Decimal("0")


def test_simulate_maturity_no_state_change(svc, tenant, term_product):
	before_accounts = dict(svc._accounts)
	svc.simulate_maturity(tenant, "TD001", Decimal("100000"), 180)
	assert svc._accounts == before_accounts


# ── Products by balance ──────────────────────────────────────────────────────

def test_get_products_by_balance(svc, tenant, savings_product):
	results = svc.get_products_by_balance(tenant, Decimal("10000"), "KES")
	assert len(results) >= 1
	assert all(p.currency == "KES" for p in results)


def test_get_products_by_balance_below_minimum(svc, tenant, savings_product):
	results = svc.get_products_by_balance(tenant, Decimal("100"), "KES")
	# savings requires 500 minimum opening
	assert not any(p.code == "SAV001" for p in results)


# ── Product stats ────────────────────────────────────────────────────────────

def test_get_product_stats(svc, tenant, savings_product, term_product, savings_account):
	stats = svc.get_product_stats(tenant)
	assert stats["total_products"] == 2
	assert stats["active_products"] == 2
	assert "KES" in stats["currencies"]
	assert stats["total_accounts"] >= 1


# ── WHT report ───────────────────────────────────────────────────────────────

def test_withholding_tax_report_monthly(svc, tenant, savings_product, savings_account):
	svc.apply_interest(tenant, "ACC-SAV-001", Decimal("1000"), date(2025, 3, 31), "P-WHT-1")
	report = svc.get_withholding_tax_report(tenant, "2025-03")
	assert len(report) >= 1
	assert all(e.wht_amount > Decimal("0") for e in report)


def test_withholding_tax_report_quarterly(svc, tenant, savings_product, savings_account):
	svc.apply_interest(tenant, "ACC-SAV-001", Decimal("3000"), date(2025, 3, 31), "P-WHT-Q1")
	report = svc.get_withholding_tax_report(tenant, "2025-Q1")
	assert len(report) >= 1


# ── Health check ─────────────────────────────────────────────────────────────

def test_health_check(svc, tenant, savings_product):
	hc = svc.health_check()
	assert hc["status"] == "ok"
	assert hc["capability"] == "fin.dep"
	assert hc["total_products"] >= 1


# ── Interest history ─────────────────────────────────────────────────────────

def test_get_interest_history(svc, tenant, savings_product, savings_account):
	svc.apply_interest(tenant, "ACC-SAV-001", Decimal("100"), date(2025, 2, 28), "HIST-1")
	svc.apply_interest(tenant, "ACC-SAV-001", Decimal("120"), date(2025, 3, 31), "HIST-2")
	history = svc.get_interest_history(tenant, "ACC-SAV-001", date(2025, 1, 1), date(2025, 12, 31))
	assert len(history) == 2


def test_get_interest_history_date_filter(svc, tenant, savings_product, savings_account):
	svc.apply_interest(tenant, "ACC-SAV-001", Decimal("100"), date(2025, 1, 31), "H1")
	svc.apply_interest(tenant, "ACC-SAV-001", Decimal("120"), date(2025, 6, 30), "H2")
	history = svc.get_interest_history(tenant, "ACC-SAV-001", date(2025, 1, 1), date(2025, 3, 31))
	assert len(history) == 1
