"""World-class payroll test suite.

Covers:
- All PAYE engines (KE, TZ, UG, GH, NG, ZA, ZM, RW)
- All statutory deductions by country
- Full payroll run orchestration
- Bonus runs, overtime, proration
- Terminal benefits / final settlement
- Leave encashment and carry-forward
- Salary advances and garnishments
- GL posting balance check
- Bank transfer file generation
- P9 form and statutory returns
- Expatriate tax equalisation
- Salary sacrifice / pension
- Domain rules (RuleViolation assertions)
- API Blueprint routes
"""
from __future__ import annotations

import asyncio
import json
from datetime import date
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _make_svc(tenant="test-tenant"):
	import sys, os
	sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
	from service import PayrollManagementService
	svc = PayrollManagementService(tenant_id=tenant, user_id="test-user")
	return svc, tenant


def _setup_pay_group(svc, tenant, country="KE", currency="KES"):
	return svc.create_pay_group("pg1", tenant, "KE-MONTHLY", "Kenya Monthly", "monthly", currency, country, "admin")


def _setup_profile(svc, tenant, pay_group_id, base_pay=100_000.0, hire_date="2020-01-01"):
	return svc.create_employee_pay_profile(
		"prof1", tenant, "emp-001", pay_group_id, "bank_transfer",
		"A012345678Z", "KES", base_pay, "test-reviewer",
		basic_pay=base_pay * 0.6, hire_date=hire_date, bank_account="1234567890",
	)


# ===========================================================================
# Domain calculations — pure functions
# ===========================================================================

class TestPayeKenya:
	def test_zero_income(self):
		from domain.calculations import calculate_paye_ke
		r = calculate_paye_ke(Decimal("0"))
		assert r.paye_amount == Decimal("0")

	def test_low_income_with_personal_relief(self):
		from domain.calculations import calculate_paye_ke
		# KES 24,000 gross: tax = 24000*10% = 2400; personal relief = 2400 → PAYE = 0
		r = calculate_paye_ke(Decimal("24000"))
		assert r.paye_amount == Decimal("0")
		assert r.personal_relief == Decimal("2400")

	def test_mid_income(self):
		from domain.calculations import calculate_paye_ke
		# 50,000 gross: bands 24000@10%=2400, 8333@25%=2083.25, 17667@30%=5300.10
		# gross_tax≈9783; minus 2400 personal relief = 7383
		r = calculate_paye_ke(Decimal("50000"))
		assert r.paye_amount > Decimal("0")
		assert r.gross_tax > r.paye_amount  # personal relief reduces it

	def test_insurance_relief_capped(self):
		from domain.calculations import calculate_paye_ke
		# 15% of 50000 = 7500, capped at 5000
		r = calculate_paye_ke(Decimal("100000"), insurance_premiums=Decimal("50000"))
		assert r.insurance_relief == Decimal("5000")

	def test_pension_deduction_reduces_taxable(self):
		from domain.calculations import calculate_paye_ke
		r_no_pension = calculate_paye_ke(Decimal("80000"))
		r_with_pension = calculate_paye_ke(Decimal("80000"), pension_employee=Decimal("4800"))
		assert r_with_pension.paye_amount < r_no_pension.paye_amount

	def test_pension_deduction_capped_at_20000(self):
		from domain.calculations import calculate_paye_ke
		r = calculate_paye_ke(Decimal("200000"), pension_employee=Decimal("50000"))
		# taxable = 200000 - 20000 (cap) = 180000
		assert r.taxable_income == Decimal("180000")

	def test_non_resident_no_personal_relief(self):
		from domain.calculations import calculate_paye_ke
		r = calculate_paye_ke(Decimal("50000"), is_resident=False)
		assert r.personal_relief == Decimal("0")

	def test_top_band_35pct(self):
		from domain.calculations import calculate_paye_ke
		r = calculate_paye_ke(Decimal("900000"))
		assert r.paye_amount > Decimal("0")
		# effective rate must be significant
		assert r.paye_amount / Decimal("900000") > Decimal("0.25")


class TestPayeTanzania:
	def test_zero_band(self):
		from domain.calculations import calculate_paye_tz
		r = calculate_paye_tz(Decimal("270000"))
		assert r.paye_amount == Decimal("0")

	def test_above_zero_band(self):
		from domain.calculations import calculate_paye_tz
		r = calculate_paye_tz(Decimal("400000"))
		# 270k@0 + 130k@8% = 10400
		assert r.paye_amount == Decimal("10400.00")

	def test_high_income(self):
		from domain.calculations import calculate_paye_tz
		r = calculate_paye_tz(Decimal("2000000"))
		assert r.paye_amount > Decimal("0")


class TestPayeUganda:
	def test_zero_band(self):
		from domain.calculations import calculate_paye_ug
		r = calculate_paye_ug(Decimal("235000"))
		assert r.paye_amount == Decimal("0")

	def test_surcharge_above_10m(self):
		from domain.calculations import calculate_paye_ug
		r = calculate_paye_ug(Decimal("11000000"))
		# must include 10% surcharge on computed tax
		r_no_surcharge = calculate_paye_ug(Decimal("10000000"))
		assert r.paye_amount > r_no_surcharge.paye_amount


class TestPayeGhana:
	def test_zero_band(self):
		from domain.calculations import calculate_paye_gh
		r = calculate_paye_gh(Decimal("402"))
		assert r.paye_amount == Decimal("0")

	def test_multi_band(self):
		from domain.calculations import calculate_paye_gh
		r = calculate_paye_gh(Decimal("5000"))
		assert r.paye_amount > Decimal("0")


class TestPayeNigeria:
	def test_annual_computation(self):
		from domain.calculations import calculate_paye_ng
		# Annual gross 3,600,000 NGN (300k/month)
		r = calculate_paye_ng(Decimal("3600000"))
		assert r.paye_amount > Decimal("0")
		# CRA = max(200k, 1%) + 20% = 200k + 720k = 920k
		assert r.personal_relief > Decimal("0")

	def test_pension_reduces_taxable(self):
		from domain.calculations import calculate_paye_ng
		r1 = calculate_paye_ng(Decimal("5000000"))
		r2 = calculate_paye_ng(Decimal("5000000"), pension_employee=Decimal("400000"))
		assert r2.paye_amount < r1.paye_amount


class TestPayeSouthAfrica:
	def test_below_threshold(self):
		from domain.calculations import calculate_paye_za
		# Annual 80,000 ZAR — below primary threshold (~91,250 effective after rebate)
		r = calculate_paye_za(Decimal("80000"))
		assert r.paye_amount == Decimal("0")

	def test_age_rebate(self):
		from domain.calculations import calculate_paye_za
		r_young = calculate_paye_za(Decimal("500000"), age=30)
		r_old = calculate_paye_za(Decimal("500000"), age=65)
		assert r_old.paye_amount < r_young.paye_amount


class TestPayeZambia:
	def test_zero_band(self):
		from domain.calculations import calculate_paye_zm
		r = calculate_paye_zm(Decimal("4800"))
		assert r.paye_amount == Decimal("0")

	def test_above_zero_band(self):
		from domain.calculations import calculate_paye_zm
		r = calculate_paye_zm(Decimal("6000"))
		# (6000-4800) * 20% = 240
		assert r.paye_amount == Decimal("240.00")


class TestPayeRwanda:
	def test_zero_band(self):
		from domain.calculations import calculate_paye_rw
		r = calculate_paye_rw(Decimal("30000"))
		assert r.paye_amount == Decimal("0")


class TestPayeDispatcher:
	def test_all_countries(self):
		from domain.calculations import calculate_paye
		for country in ["KE", "TZ", "UG", "GH", "NG", "ZA", "ZM", "RW"]:
			r = calculate_paye(country, Decimal("100000"))
			assert r.paye_amount >= Decimal("0"), f"negative PAYE for {country}"

	def test_unknown_country_flat_30pct(self):
		from domain.calculations import calculate_paye
		r = calculate_paye("XX", Decimal("100000"))
		assert r.paye_amount == Decimal("30000.00")


# ===========================================================================
# Statutory deductions
# ===========================================================================

class TestStatutoryDeductions:
	def test_ke_nssf(self):
		from domain.calculations import calculate_nssf_ke
		r = calculate_nssf_ke(Decimal("50000"))
		# capped at 18000 * 6% = 1080
		assert r.employee_amount == Decimal("1080.00")
		assert r.employer_amount == Decimal("1080.00")
		assert r.cap_applied

	def test_ke_nssf_under_cap(self):
		from domain.calculations import calculate_nssf_ke
		r = calculate_nssf_ke(Decimal("10000"))
		assert r.employee_amount == Decimal("600.00")
		assert not r.cap_applied

	def test_ke_nhif_shif(self):
		from domain.calculations import calculate_nhif_ke
		r = calculate_nhif_ke(Decimal("50000"))
		# 2.75% of 50000 = 1375
		assert r.employee_amount == Decimal("1375.00")

	def test_ke_nhif_minimum(self):
		from domain.calculations import calculate_nhif_ke
		r = calculate_nhif_ke(Decimal("5000"))
		# 2.75% of 5000 = 137.5 < 300 minimum
		assert r.employee_amount == Decimal("300")

	def test_ke_nita(self):
		from domain.calculations import calculate_nita_ke
		r = calculate_nita_ke()
		assert r.employer_amount == Decimal("50")
		assert r.employee_amount == Decimal("0")

	def test_tz_nssf(self):
		from domain.calculations import calculate_nssf_tz
		r = calculate_nssf_tz(Decimal("1000000"))
		assert r.employee_amount == Decimal("100000.00")
		assert r.employer_amount == Decimal("100000.00")

	def test_tz_sdl(self):
		from domain.calculations import calculate_sdl_tz
		r = calculate_sdl_tz(Decimal("1000000"))
		assert r.employer_amount == Decimal("40000.00")

	def test_tz_wcf_capped(self):
		from domain.calculations import calculate_wcf_tz
		r = calculate_wcf_tz(Decimal("1000000"))
		# 0.5% of 1M = 5000 > cap 3000
		assert r.employer_amount == Decimal("3000")
		assert r.cap_applied

	def test_gh_ssnit(self):
		from domain.calculations import calculate_ssnit_gh
		r = calculate_ssnit_gh(Decimal("5000"))
		assert r.employee_amount == Decimal("275.00")  # 5.5%
		assert r.employer_amount == Decimal("650.00")  # 13%

	def test_ng_pencom(self):
		from domain.calculations import calculate_pencom_ng
		r = calculate_pencom_ng(Decimal("300000"))
		assert r.employee_amount == Decimal("24000.00")  # 8%
		assert r.employer_amount == Decimal("30000.00")  # 10%

	def test_zm_napsa_capped(self):
		from domain.calculations import calculate_napsa_zm
		r = calculate_napsa_zm(Decimal("100000"))
		# 5% of 100000 = 5000 > cap 1221.80
		assert r.employee_amount == Decimal("1221.80")
		assert r.cap_applied

	def test_za_uif_capped(self):
		from domain.calculations import calculate_uif_za
		r = calculate_uif_za(Decimal("50000"))
		# 1% of 50000 = 500 > cap 177.12
		assert r.employee_amount == Decimal("177.12")
		assert r.cap_applied

	def test_dispatcher_ke(self):
		from domain.calculations import calculate_statutory_deductions
		results = calculate_statutory_deductions("KE", Decimal("60000"))
		types = {r.deduction_type for r in results}
		assert "nssf" in types
		assert "nhif_shi" in types
		assert "nita" in types

	def test_dispatcher_tz(self):
		from domain.calculations import calculate_statutory_deductions
		results = calculate_statutory_deductions("TZ", Decimal("800000"))
		types = {r.deduction_type for r in results}
		assert "nssf" in types
		assert "sdl" in types


# ===========================================================================
# Domain calculations — proration, overtime, terminal benefits
# ===========================================================================

class TestProration:
	def test_full_month(self):
		from domain.calculations import prorate_salary
		result = prorate_salary(Decimal("100000"), 22, 22)
		assert result == Decimal("100000.00")

	def test_partial_month(self):
		from domain.calculations import prorate_salary
		result = prorate_salary(Decimal("100000"), 22, 11)
		assert result == Decimal("50000.00")

	def test_zero_days(self):
		from domain.calculations import prorate_salary
		result = prorate_salary(Decimal("100000"), 22, 0)
		assert result == Decimal("0")

	def test_working_days(self):
		from domain.calculations import working_days
		# Mon-Fri in a 5-day week
		days = working_days(date(2026, 6, 1), date(2026, 6, 5))
		assert days == 5

	def test_working_days_excludes_weekends(self):
		from domain.calculations import working_days
		# Full week Jun 1-7 (Mon-Sun) = 5 working days
		days = working_days(date(2026, 6, 1), date(2026, 6, 7))
		assert days == 5


class TestOvertimeCalculation:
	def test_time_and_half(self):
		from domain.calculations import calculate_overtime_amount
		result = calculate_overtime_amount(Decimal("100000"), 173, Decimal("10"), Decimal("1.5"))
		# hourly = 100000/173 ≈ 578.03; OT rate = 578.03*1.5 ≈ 867.05; 10h = 8670.xx
		assert result > Decimal("8000")
		assert result < Decimal("9500")

	def test_zero_hours(self):
		from domain.calculations import calculate_overtime_amount
		result = calculate_overtime_amount(Decimal("100000"), 173, Decimal("0"))
		assert result == Decimal("0")


class TestTerminalBenefits:
	def test_ke_severance_5_years(self):
		from domain.calculations import calculate_severance_pay_ke
		result = calculate_severance_pay_ke(Decimal("100000"), 5)
		# 5 * 15/26 * 100000
		expected = (Decimal("100000") / Decimal("26") * Decimal("15") * Decimal("5")).quantize(Decimal("0.01"))
		assert result == expected

	def test_ke_severance_less_than_1_year(self):
		from domain.calculations import calculate_severance_pay_ke
		result = calculate_severance_pay_ke(Decimal("100000"), 0)
		assert result == Decimal("0")

	def test_notice_pay(self):
		from domain.calculations import calculate_notice_pay
		result = calculate_notice_pay(Decimal("100000"), 30, 22)
		expected = (Decimal("100000") / 22 * 30).quantize(Decimal("0.01"))
		assert result == expected

	def test_gratuity(self):
		from domain.calculations import calculate_gratuity
		result = calculate_gratuity(Decimal("100000"), Decimal("3"), Decimal("0.25"))
		# 100000 * 12 * 3 * 0.25 = 900000
		assert result == Decimal("900000.00")

	def test_leave_encashment(self):
		from domain.calculations import daily_rate, encash_leave
		dr = daily_rate(Decimal("100000"), 22)
		result = encash_leave(dr, Decimal("10"))
		assert result > Decimal("0")


class TestExpatTax:
	def test_non_resident_below_183(self):
		from domain.calculations import assess_expat_tax
		r = assess_expat_tax(Decimal("100000"), days_in_country=100)
		assert not r.is_tax_resident
		assert r.taxable_income == Decimal("0")

	def test_resident_above_183(self):
		from domain.calculations import assess_expat_tax
		r = assess_expat_tax(Decimal("100000"), days_in_country=200, host_country_paye=Decimal("30000"))
		assert r.is_tax_resident
		assert r.estimated_tax == Decimal("30000")

	def test_tax_equalisation(self):
		from domain.calculations import assess_expat_tax
		r = assess_expat_tax(
			Decimal("100000"),
			days_in_country=200,
			home_country_tax_rate=Decimal("0.40"),
			has_tax_equalisation=True,
		)
		assert r.effective_rate == Decimal("0.40")


class TestVariance:
	def test_positive_variance(self):
		from domain.calculations import compute_variance_pct
		r = compute_variance_pct(Decimal("110"), Decimal("100"))
		assert r == Decimal("10.00")

	def test_zero_previous(self):
		from domain.calculations import compute_variance_pct
		assert compute_variance_pct(Decimal("100"), Decimal("0")) is None

	def test_negative_variance(self):
		from domain.calculations import compute_variance_pct
		r = compute_variance_pct(Decimal("90"), Decimal("100"))
		assert r == Decimal("-10.00")


# ===========================================================================
# Domain rules
# ===========================================================================

class TestDomainRules:
	def test_tenant_match_passes(self):
		from domain.rules import assert_tenant_match
		assert_tenant_match("t1", "t1")  # no exception

	def test_tenant_match_fails(self):
		from domain.rules import assert_tenant_match, RuleViolation
		with pytest.raises(RuleViolation) as exc_info:
			assert_tenant_match("t1", "t2")
		assert "PR-001" in str(exc_info.value)

	def test_period_dates_valid(self):
		from domain.rules import assert_period_dates_valid
		assert_period_dates_valid(date(2026, 1, 1), date(2026, 1, 31), date(2026, 2, 5))

	def test_period_end_before_start_fails(self):
		from domain.rules import assert_period_dates_valid, RuleViolation
		with pytest.raises(RuleViolation):
			assert_period_dates_valid(date(2026, 1, 31), date(2026, 1, 1), date(2026, 2, 5))

	def test_run_status_check(self):
		from domain.rules import assert_run_in_status, RuleViolation
		run = {"id": "r1", "status": "draft"}
		assert_run_in_status(run, "draft", "calculated")
		with pytest.raises(RuleViolation):
			assert_run_in_status(run, "posted")

	def test_journal_balanced(self):
		from domain.rules import assert_journal_balanced
		assert_journal_balanced(Decimal("100"), Decimal("100"))

	def test_journal_unbalanced_fails(self):
		from domain.rules import assert_journal_balanced, RuleViolation
		with pytest.raises(RuleViolation):
			assert_journal_balanced(Decimal("100"), Decimal("99"))

	def test_leave_type_encashable(self):
		from domain.rules import assert_leave_type_encashable, RuleViolation
		assert_leave_type_encashable("annual")
		with pytest.raises(RuleViolation):
			assert_leave_type_encashable("sick")

	def test_advance_within_limit(self):
		from domain.rules import assert_advance_within_limit, RuleViolation
		assert_advance_within_limit(Decimal("300000"), Decimal("100000"), 3)
		with pytest.raises(RuleViolation):
			assert_advance_within_limit(Decimal("400000"), Decimal("100000"), 3)

	def test_prorated_salary(self):
		from domain.rules import calculate_prorated_salary
		result = calculate_prorated_salary(Decimal("100000"), 30, 15)
		assert result == Decimal("50000.00")

	def test_calculate_days_worked_mid_hire(self):
		from domain.rules import calculate_days_worked
		days_worked, days_in_period = calculate_days_worked(
			hire_date=date(2026, 6, 15),
			period_start=date(2026, 6, 1),
			period_end=date(2026, 6, 30),
		)
		assert days_worked == 16
		assert days_in_period == 30

	def test_garnishment_cap(self):
		from domain.rules import assert_garnishment_within_legal_limit
		# disposable 90000; requested 35000 > 33.33% cap = 29997
		capped = assert_garnishment_within_legal_limit(Decimal("90000"), Decimal("35000"))
		assert capped < Decimal("35000")


# ===========================================================================
# Service — async methods
# ===========================================================================

class TestPayrollService:
	def test_calculate_paye_ke(self):
		svc, tenant = _make_svc()
		result = run(svc.calculate_paye(100_000, "KE"))
		assert result["paye_payable"] > 0
		assert result["country"] == "KE"

	def test_calculate_paye_ng_annual(self):
		svc, tenant = _make_svc()
		result = run(svc.calculate_paye(300_000, "NG"))
		assert result["paye_payable"] >= 0

	def test_calculate_paye_unsupported_country(self):
		from service import CountryNotSupportedError
		svc, tenant = _make_svc()
		with pytest.raises(CountryNotSupportedError):
			run(svc.calculate_paye(100_000, "XX"))

	def test_calculate_statutory_ke(self):
		svc, tenant = _make_svc()
		result = run(svc.calculate_statutory_deductions({}, 100_000, "KE"))
		assert result["ee_total"] > 0
		assert result["er_total"] > 0
		names = [b["name"] for b in result["breakdown"]]
		assert "NSSF" in names
		assert "NHIF" in names

	def test_mid_month_hire(self):
		svc, tenant = _make_svc()
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 100_000, "2026-06-15")
		result = run(svc.mid_month_hire_calculation("emp-001", "2026-06-15", "2026-06-01", tenant_id=tenant))
		assert result["days_worked"] == 16
		assert result["prorated_pay"] < 100_000

	def test_full_payroll_run(self):
		svc, tenant = _make_svc()
		period = svc.create_payroll_period("p1", tenant, "June 2026", "monthly", "2026-06-01", "2026-06-30", "2026-07-01", "KES")
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 150_000)
		result = run(svc.run_payroll(period["id"], tenant, pg["id"], "admin"))
		assert result["employee_count"] == 1
		assert result["totals"]["gross"] == 150_000.0
		assert result["totals"]["net"] > 0
		assert len(result["payslip_lines"]) == 1

	def test_run_net_less_than_gross(self):
		svc, tenant = _make_svc()
		period = svc.create_payroll_period("p1", tenant, "June 2026", "monthly", "2026-06-01", "2026-06-30", "2026-07-01", "KES")
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 80_000)
		result = run(svc.run_payroll(period["id"], tenant, pg["id"], "admin"))
		net = result["totals"]["net"]
		gross = result["totals"]["gross"]
		assert net < gross

	def test_bonus_run_aggregate(self):
		svc, tenant = _make_svc()
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 100_000)
		result = run(svc.process_bonus_payroll(
			"annual", ["emp-001"], {"emp-001": 50_000}, "aggregate", tenant_id=tenant,
		))
		assert result["employee_count"] == 1
		assert result["totals"]["gross"] == 50_000.0
		assert result["totals"]["net"] < 50_000.0

	def test_bonus_run_separate_rate(self):
		svc, tenant = _make_svc()
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 100_000)
		result = run(svc.process_bonus_payroll(
			"spot", ["emp-001"], {"emp-001": 20_000}, "separate_rate", tenant_id=tenant,
		))
		# 30% flat
		assert result["lines"][0]["paye"] == round(20_000 * 0.30, 2)

	def test_overtime_time_and_half(self):
		svc, tenant = _make_svc()
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 86_666.5)
		result = run(svc.calculate_overtime("emp-001", 173.33, 10, "time_and_half", tenant_id=tenant))
		assert result["overtime_multiplier"] == 1.5
		assert result["overtime_pay"] > 0

	def test_overtime_invalid_type(self):
		from service import PayrollError
		svc, tenant = _make_svc()
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"])
		with pytest.raises(PayrollError):
			run(svc.calculate_overtime("emp-001", 173, 10, "invalid_type", tenant_id=tenant))

	def test_terminal_benefits_redundancy(self):
		svc, tenant = _make_svc()
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 100_000, "2019-01-01")
		result = run(svc.calculate_terminal_benefits("emp-001", "2026-06-01", "redundancy", tenant_id=tenant, leave_days_accrued=15))
		assert result["severance_pay"] > 0  # 7 years served
		assert result["leave_encashment"] > 0
		assert result["exempt_terminal"] > 0  # redundancy is tax-exempt

	def test_terminal_benefits_resignation(self):
		svc, tenant = _make_svc()
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 100_000, "2020-01-01")
		result = run(svc.calculate_terminal_benefits("emp-001", "2026-06-01", "resignation", tenant_id=tenant))
		# No severance for resignation
		assert result["severance_pay"] == 0.0

	def test_leave_encashment_annual(self):
		svc, tenant = _make_svc()
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 100_000)
		result = run(svc.calculate_leave_encashment("emp-001", "annual", 10, tenant_id=tenant))
		assert result["encashment_gross"] > 0

	def test_leave_encashment_sick_fails(self):
		from service import PayrollError
		svc, tenant = _make_svc()
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"])
		with pytest.raises(PayrollError):
			run(svc.calculate_leave_encashment("emp-001", "sick", 5, tenant_id=tenant))

	def test_generate_payslip(self):
		svc, tenant = _make_svc()
		period = svc.create_payroll_period("p1", tenant, "June 2026", "monthly", "2026-06-01", "2026-06-30", "2026-07-01", "KES")
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 120_000)
		run_record = run(svc.run_payroll(period["id"], tenant, pg["id"], "admin"))
		payslip = run(svc.generate_payslip("emp-001", run_record["id"], tenant_id=tenant))
		assert payslip["net_pay"] > 0
		assert payslip["employee_id"] == "emp-001"
		assert payslip["paye"] > 0

	def test_p9_form(self):
		svc, tenant = _make_svc()
		period = svc.create_payroll_period("p1", tenant, "Jan 2026", "monthly", "2026-01-01", "2026-01-31", "2026-02-01", "KES")
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 80_000)
		run_record = run(svc.run_payroll(period["id"], tenant, pg["id"], "admin"))
		svc.runs[run_record["id"]]["status"] = "posted"
		p9 = run(svc.generate_p9_form("emp-001", 2026, tenant_id=tenant))
		assert p9["year"] == 2026
		assert p9["authority"] == "KRA"

	def test_gl_posting_balanced(self):
		svc, tenant = _make_svc()
		period = svc.create_payroll_period("p1", tenant, "June 2026", "monthly", "2026-06-01", "2026-06-30", "2026-07-01", "KES")
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 100_000)
		run_record = run(svc.run_payroll(period["id"], tenant, pg["id"], "admin"))
		svc.runs[run_record["id"]]["status"] = "posted"
		gl = run(svc.gl_posting(run_record["id"], tenant_id=tenant))
		assert gl["balanced"] is True
		assert abs(gl["total_debits"] - gl["total_credits"]) < 0.02

	def test_bank_transfer_file(self):
		svc, tenant = _make_svc()
		period = svc.create_payroll_period("p1", tenant, "June 2026", "monthly", "2026-06-01", "2026-06-30", "2026-07-01", "KES")
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 90_000)
		run_record = run(svc.run_payroll(period["id"], tenant, pg["id"], "admin"))
		svc.runs[run_record["id"]]["status"] = "posted"
		result = run(svc.bank_transfer_file(run_record["id"], "KCB_EFT", tenant_id=tenant))
		assert result["record_count"] == 1
		assert result["total_amount"] > 0
		assert "account_number" in result["file_content"]

	def test_payroll_variance_report(self):
		svc, tenant = _make_svc()
		period1 = svc.create_payroll_period("p1", tenant, "May 2026", "monthly", "2026-05-01", "2026-05-31", "2026-06-01", "KES")
		period2 = svc.create_payroll_period("p2", tenant, "Jun 2026", "monthly", "2026-06-01", "2026-06-30", "2026-07-01", "KES")
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 100_000)
		run1 = run(svc.run_payroll(period1["id"], tenant, pg["id"], "admin"))
		run2 = run(svc.run_payroll(period2["id"], tenant, pg["id"], "admin"))
		svc.runs[run1["id"]]["status"] = "posted"
		svc.runs[run2["id"]]["status"] = "posted"
		report = run(svc.payroll_variance_report(run2["id"], tenant_id=tenant))
		assert "variances" in report
		assert "generated_at" in report

	def test_statutory_returns(self):
		svc, tenant = _make_svc()
		period = svc.create_payroll_period("p1", tenant, "June 2026", "monthly", "2026-06-01", "2026-06-30", "2026-07-01", "KES")
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 100_000)
		run_record = run(svc.run_payroll(period["id"], tenant, pg["id"], "admin"))
		svc.runs[run_record["id"]]["status"] = "posted"
		returns = run(svc.generate_statutory_returns(period["id"], "KE", tenant_id=tenant))
		assert "nssf_schedule" in returns
		assert "nhif_schedule" in returns
		assert "paye_schedule" in returns

	def test_salary_advance_create_and_deduct(self):
		svc, tenant = _make_svc()
		period = svc.create_payroll_period("p1", tenant, "June 2026", "monthly", "2026-06-01", "2026-06-30", "2026-07-01", "KES")
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 100_000)
		run_record = run(svc.run_payroll(period["id"], tenant, pg["id"], "admin"))
		advance = svc.create_salary_advance("adv1", tenant, "emp-001", 30_000.0, 10_000.0, "hr-manager")
		result = run(svc.apply_salary_advance_deduction("emp-001", advance["id"], run_record["id"], tenant_id=tenant))
		assert result["deducted_amount"] == 10_000.0
		assert result["remaining_balance"] == 20_000.0

	def test_garnishment_capped(self):
		svc, tenant = _make_svc()
		period = svc.create_payroll_period("p1", tenant, "June 2026", "monthly", "2026-06-01", "2026-06-30", "2026-07-01", "KES")
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 100_000)
		run_record = run(svc.run_payroll(period["id"], tenant, pg["id"], "admin"))
		order = {
			"order_id": "COURT-123",
			"creditor": "Bank Ltd",
			"amount_or_pct": 80_000,  # exceeds 33.33% of net
			"order_type": "fixed",
			"max_pct": 33.33,
		}
		result = run(svc.process_garnishment("emp-001", order, run_record["id"], tenant_id=tenant))
		assert result["capped"] is True
		assert result["garnishment_amount"] < 80_000

	def test_expatriate_tax_equalisation(self):
		svc, tenant = _make_svc()
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 500_000)
		result = run(svc.expatriate_tax_calculation("emp-001", "2026-06", tenant_id=tenant, home_country="GB", host_country="KE", company_bearing_tax=True))
		assert "company_tax_cost" in result
		assert "net_to_employee" in result

	def test_salary_sacrifice_pension(self):
		svc, tenant = _make_svc()
		pg = _setup_pay_group(svc, tenant)
		_setup_profile(svc, tenant, pg["id"], 100_000)
		result = run(svc.salary_sacrifice_pension("emp-001", 10_000.0, tenant_id=tenant, is_percentage=False))
		assert result["paye_saving"] > 0
		assert result["paye_after"] < result["paye_before"]


# ===========================================================================
# API Blueprint (Flask test client)
# ===========================================================================

class TestPayrollApi:
	def _client(self):
		from flask import Flask
		import sys, os
		sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
		from api import bp
		app = Flask(__name__)
		app.register_blueprint(bp)
		app.config["TESTING"] = True
		return app.test_client()

	def _headers(self):
		return {"X-Tenant-Id": "api-test-tenant", "Content-Type": "application/json"}

	def test_health(self):
		client = self._client()
		resp = client.get("/api/v1/payroll/health", headers=self._headers())
		assert resp.status_code == 200
		data = resp.get_json()
		assert data["data"]["status"] == "ok"

	def test_dashboard(self):
		client = self._client()
		resp = client.get("/api/v1/payroll/dashboard", headers=self._headers())
		assert resp.status_code == 200

	def test_create_period(self):
		client = self._client()
		payload = {
			"period_code": "2026-06",
			"name": "June 2026",
			"pay_frequency": "monthly",
			"start_date": "2026-06-01",
			"end_date": "2026-06-30",
			"pay_date": "2026-07-01",
			"currency": "KES",
		}
		resp = client.post("/api/v1/payroll/periods", json=payload, headers=self._headers())
		assert resp.status_code == 201
		data = resp.get_json()
		assert data["data"]["status"] == "open"

	def test_list_periods(self):
		client = self._client()
		resp = client.get("/api/v1/payroll/periods", headers=self._headers())
		assert resp.status_code == 200
		data = resp.get_json()
		assert "data" in data
		assert "meta" in data

	def test_create_pay_group(self):
		client = self._client()
		payload = {"code": "KE-M", "name": "Kenya Monthly", "country": "KE", "currency": "KES"}
		resp = client.post("/api/v1/payroll/pay-groups", json=payload, headers=self._headers())
		assert resp.status_code == 201

	def test_calculate_paye_endpoint(self):
		client = self._client()
		payload = {"gross_monthly": 100000, "country": "KE"}
		resp = client.post("/api/v1/payroll/tax/calculate-paye", json=payload, headers=self._headers())
		assert resp.status_code == 200
		data = resp.get_json()
		assert data["data"]["paye_payable"] > 0

	def test_calculate_statutory_endpoint(self):
		client = self._client()
		payload = {"gross": 80000, "country": "KE"}
		resp = client.post("/api/v1/payroll/tax/calculate-statutory", json=payload, headers=self._headers())
		assert resp.status_code == 200
		data = resp.get_json()
		assert data["data"]["ee_total"] > 0

	def test_create_employee(self):
		client = self._client()
		payload = {
			"employee_number": "EMP-001",
			"full_name": "Alice Wanjiku",
			"national_id": "12345678",
			"tax_pin": "A012345678Z",
			"hire_date": "2022-03-01",
			"basic_salary": 85000,
			"currency": "KES",
			"country": "KE",
		}
		resp = client.post("/api/v1/payroll/employees", json=payload, headers=self._headers())
		assert resp.status_code == 201
		data = resp.get_json()
		assert data["data"]["full_name"] == "Alice Wanjiku"

	def test_create_profile(self):
		client = self._client()
		# First create pay group
		pg_resp = client.post("/api/v1/payroll/pay-groups",
			json={"code": "KE-M", "name": "Kenya Monthly", "country": "KE", "currency": "KES"},
			headers=self._headers())
		pg_id = pg_resp.get_json()["data"]["id"]
		payload = {
			"employee_id": "emp-test-001",
			"pay_group_id": pg_id,
			"base_pay": 100000,
			"tax_id": "A012345678Z",
			"reviewed_by": "hr-manager",
		}
		resp = client.post("/api/v1/payroll/profiles", json=payload, headers=self._headers())
		assert resp.status_code == 201

	def test_run_lifecycle(self):
		client = self._client()
		headers = self._headers()
		# Create period
		p_resp = client.post("/api/v1/payroll/periods",
			json={"period_code": "t-2026-06", "start_date": "2026-06-01", "end_date": "2026-06-30", "pay_date": "2026-07-01"},
			headers=headers)
		period_id = p_resp.get_json()["data"]["id"]
		# Create pay group
		pg_resp = client.post("/api/v1/payroll/pay-groups",
			json={"code": "T-KE", "name": "Test KE", "country": "KE", "currency": "KES"},
			headers=headers)
		pg_id = pg_resp.get_json()["data"]["id"]
		# Create run
		run_resp = client.post("/api/v1/payroll/runs",
			json={"period_id": period_id, "pay_group_id": pg_id, "initiated_by": "test-user"},
			headers=headers)
		assert run_resp.status_code == 201
		run_id = run_resp.get_json()["data"]["id"]
		# Approve
		approve_resp = client.post(f"/api/v1/payroll/runs/{run_id}/approve",
			json={"approved_by": "cfo"}, headers=headers)
		assert approve_resp.status_code == 200
		assert approve_resp.get_json()["data"]["status"] == "approved"
		# Post
		post_resp = client.post(f"/api/v1/payroll/runs/{run_id}/post",
			json={"posted_by": "cfo"}, headers=headers)
		assert post_resp.status_code == 200
		assert post_resp.get_json()["data"]["status"] == "posted"

	def test_midmonth_proration_endpoint(self):
		client = self._client()
		# need a profile
		pg_resp = client.post("/api/v1/payroll/pay-groups",
			json={"code": "PR-KE", "name": "Prorate KE", "country": "KE", "currency": "KES"},
			headers=self._headers())
		pg_id = pg_resp.get_json()["data"]["id"]
		client.post("/api/v1/payroll/profiles",
			json={"employee_id": "prorate-emp", "pay_group_id": pg_id, "base_pay": 120000, "tax_id": "A000000000Z", "reviewed_by": "hr"},
			headers=self._headers())
		payload = {"employee_id": "prorate-emp", "hire_date": "2026-06-15", "period": "2026-06-01"}
		resp = client.post("/api/v1/payroll/proration/mid-hire", json=payload, headers=self._headers())
		assert resp.status_code == 200
		data = resp.get_json()["data"]
		assert data["days_worked"] == 16

	def test_bonus_run_endpoint(self):
		client = self._client()
		pg_resp = client.post("/api/v1/payroll/pay-groups",
			json={"code": "BN-KE", "name": "Bonus KE", "country": "KE", "currency": "KES"},
			headers=self._headers())
		pg_id = pg_resp.get_json()["data"]["id"]
		client.post("/api/v1/payroll/profiles",
			json={"employee_id": "bonus-emp-001", "pay_group_id": pg_id, "base_pay": 100000, "tax_id": "B000000001Z", "reviewed_by": "hr"},
			headers=self._headers())
		payload = {
			"bonus_type": "annual",
			"employee_ids": ["bonus-emp-001"],
			"amounts": {"bonus-emp-001": 50000},
			"tax_method": "aggregate",
		}
		resp = client.post("/api/v1/payroll/runs/bonus", json=payload, headers=self._headers())
		assert resp.status_code == 201
		data = resp.get_json()["data"]
		assert data["totals"]["gross"] == 50000.0

	def test_expatriate_tax_endpoint(self):
		client = self._client()
		pg_resp = client.post("/api/v1/payroll/pay-groups",
			json={"code": "EX-KE", "name": "Expat KE", "country": "KE", "currency": "KES"},
			headers=self._headers())
		pg_id = pg_resp.get_json()["data"]["id"]
		client.post("/api/v1/payroll/profiles",
			json={"employee_id": "expat-001", "pay_group_id": pg_id, "base_pay": 500000, "tax_id": "C000000001Z", "reviewed_by": "hr"},
			headers=self._headers())
		payload = {"employee_id": "expat-001", "home_country": "GB", "host_country": "KE"}
		resp = client.post("/api/v1/payroll/tax/expatriate", json=payload, headers=self._headers())
		assert resp.status_code == 200
		data = resp.get_json()["data"]
		assert "company_tax_cost" in data

	def test_salary_sacrifice_endpoint(self):
		client = self._client()
		pg_resp = client.post("/api/v1/payroll/pay-groups",
			json={"code": "SS-KE", "name": "SalSac KE", "country": "KE", "currency": "KES"},
			headers=self._headers())
		pg_id = pg_resp.get_json()["data"]["id"]
		client.post("/api/v1/payroll/profiles",
			json={"employee_id": "salsac-001", "pay_group_id": pg_id, "base_pay": 100000, "tax_id": "D000000001Z", "reviewed_by": "hr"},
			headers=self._headers())
		payload = {"employee_id": "salsac-001", "amount_or_pct": 10000, "is_percentage": False}
		resp = client.post("/api/v1/payroll/tax/salary-sacrifice", json=payload, headers=self._headers())
		assert resp.status_code == 200
		data = resp.get_json()["data"]
		assert data["paye_saving"] > 0

	def test_audit_events(self):
		client = self._client()
		resp = client.get("/api/v1/payroll/audit", headers=self._headers())
		assert resp.status_code == 200

	def test_404_run_not_found(self):
		client = self._client()
		resp = client.get("/api/v1/payroll/runs/nonexistent-run", headers=self._headers())
		assert resp.status_code == 404
