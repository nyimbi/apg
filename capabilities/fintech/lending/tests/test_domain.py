"""
Domain rules and calculations tests for APG Digital Lending.

Run: cd capabilities/fintech/lending && python -m pytest tests/test_domain.py -vxs
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
from datetime import date


# ---------------------------------------------------------------------------
# domain/calculations.py
# ---------------------------------------------------------------------------

from domain.calculations import (
	emi, flat_rate_emi, flat_rate_to_apr,
	build_amortisation_schedule,
	early_settlement_amount,
	risk_grade, probability_of_default,
	composite_credit_score, behavioural_score_raw, bureau_score_raw,
	debt_service_ratio, max_loan_from_dsr,
	ecl_stage1, ecl_stage2, ecl_stage3, lgd_from_collateral, pd_lifetime,
	forced_sale_value, collateral_coverage_ratio,
	dpd_bucket, ifrs9_stage, par_ratio,
	risk_adjusted_rate, grade_amount_cap,
	late_payment_penalty, processing_fee,
	portfolio_at_risk, stress_scenario,
)


class TestEMI:
	def test_standard(self):
		# Known: 100k at 18%/12 for 12 months
		monthly_rate = 0.18 / 12
		result = emi(100_000, monthly_rate, 12)
		assert 9100 <= result <= 9200

	def test_zero_rate(self):
		result = emi(120_000, 0.0, 12)
		assert result == 10_000.0

	def test_single_period(self):
		result = emi(100_000, 0.01, 1)
		assert result == pytest.approx(101_000.0, rel=0.01)

	def test_invalid_principal(self):
		with pytest.raises(ValueError):
			emi(-100, 0.01, 12)

	def test_invalid_n_months(self):
		with pytest.raises(ValueError):
			emi(100_000, 0.01, 0)


class TestFlatRateEMI:
	def test_standard(self):
		# 100k, 18% flat, 12 months: total interest = 18k, EMI = 9.833k
		result = flat_rate_emi(100_000, 0.18, 12)
		assert result == pytest.approx(9_833.33, rel=0.01)

	def test_flat_rate_to_apr_conversion(self):
		# 18% flat rate converts to ~32% APR for a 12-month loan
		apr = flat_rate_to_apr(0.18, 12)
		assert apr > 0.18   # APR always higher than flat rate
		assert apr < 0.45   # sanity upper bound


class TestAmortisationSchedule:
	def test_reducing_balance_length(self):
		sched = build_amortisation_schedule(100_000, 0.18, 12, date(2026, 1, 1))
		assert len(sched["installments"]) == 12

	def test_reducing_balance_zero_closing(self):
		sched = build_amortisation_schedule(100_000, 0.18, 12, date(2026, 1, 1))
		last = sched["installments"][-1]
		assert abs(last["closing_balance"]) < 1.0

	def test_total_repayable_consistent(self):
		sched = build_amortisation_schedule(100_000, 0.18, 12, date(2026, 1, 1))
		total = sum(i["emi"] for i in sched["installments"])
		assert abs(total - sched["total_repayable"]) < 1.0

	def test_flat_rate_schedule(self):
		sched = build_amortisation_schedule(100_000, 0.18, 12, date(2026, 1, 1), "flat_rate")
		assert sched["schedule_type"] == "flat_rate"
		# All EMIs equal for flat rate
		emis = [i["emi"] for i in sched["installments"]]
		assert max(emis) - min(emis) < 0.10  # within rounding

	def test_bullet_schedule(self):
		# bullet: 12 monthly interest installments + full principal in last period
		sched = build_amortisation_schedule(100_000, 0.18, 12, date(2026, 1, 1), "bullet")
		assert len(sched["installments"]) == 12
		# Last installment repays all principal
		last = sched["installments"][-1]
		assert last["principal_portion"] == pytest.approx(100_000, rel=0.01)

	def test_quarterly_frequency(self):
		sched = build_amortisation_schedule(100_000, 0.18, 12, date(2026, 1, 1), repayment_frequency="quarterly")
		# 12 months / 3 = 4 installments
		assert len(sched["installments"]) == 4

	def test_weekly_frequency(self):
		sched = build_amortisation_schedule(50_000, 0.18, 3, date(2026, 1, 1), repayment_frequency="weekly")
		assert len(sched["installments"]) == 12  # 3 months * 4 weeks


class TestEarlySettlement:
	def test_settlement_amount(self):
		result = early_settlement_amount(
			outstanding_principal=80_000,
			annual_rate=0.18,
			disbursement_date=date(2026, 1, 1),
			settlement_date=date(2026, 6, 1),
			early_settlement_fee_pct=0.01,
		)
		assert result["total_settlement_amount"] > 80_000
		assert result["accrued_interest"] > 0
		assert result["early_settlement_fee"] == pytest.approx(800.0, rel=0.01)

	def test_same_date_invalid(self):
		with pytest.raises(ValueError):
			early_settlement_amount(80_000, 0.18, date(2026, 1, 1), date(2026, 1, 1))


class TestCreditScoring:
	def test_risk_grade_thresholds(self):
		assert risk_grade(800) == "A"
		assert risk_grade(700) == "B"
		assert risk_grade(640) == "C"
		assert risk_grade(580) == "D"
		assert risk_grade(500) == "E"
		assert risk_grade(400) == "F"

	def test_pd_range(self):
		pd_a = probability_of_default(800)
		pd_f = probability_of_default(300)
		assert pd_a < 0.01   # Grade A < 1%
		assert pd_f > 0.20   # Grade F > 20%
		assert pd_a < pd_f

	def test_pd_invalid_score(self):
		with pytest.raises(ValueError):
			probability_of_default(200)

	def test_composite_score_bounds(self):
		score = composite_credit_score(0.8, 0.7, 0.9)
		assert 300 <= score <= 850

	def test_composite_score_low(self):
		score = composite_credit_score(0.0, 0.0, 0.0)
		assert score == 300

	def test_composite_score_high(self):
		score = composite_credit_score(1.0, 1.0, 1.0)
		assert score == 850

	def test_behavioural_score_raw(self):
		score = behavioural_score_raw(0.95, 0.30, 0)
		assert 0 <= score <= 1
		score_bad = behavioural_score_raw(0.40, 0.90, 5)
		assert score_bad < score

	def test_bureau_score_raw(self):
		score = bureau_score_raw(750, defaults_count=0, fraud_flags_count=0)
		assert score > 0.5
		score_bad = bureau_score_raw(400, defaults_count=3, fraud_flags_count=1)
		assert score_bad < score


class TestDSR:
	def test_passing_dsr(self):
		result = debt_service_ratio(100_000, 10_000, 20_000)
		assert result["dsr"] == pytest.approx(0.30, rel=0.01)
		assert result["passes"] is True

	def test_failing_dsr(self):
		result = debt_service_ratio(50_000, 15_000, 10_000)
		assert result["dsr"] == pytest.approx(0.50, rel=0.01)
		assert result["passes"] is False

	def test_zero_income(self):
		result = debt_service_ratio(0, 0, 1_000)
		assert result["passes"] is False
		assert math.isinf(result["dsr"])

	def test_max_loan_from_dsr(self):
		max_p = max_loan_from_dsr(100_000, 5_000, 0.18, 24)
		assert max_p > 0
		# EMI of max_p should be ≤ 40% * income - existing obligations
		affordable = 100_000 * 0.40 - 5_000
		monthly = 0.18 / 12
		factor = (1 - (1 + monthly) ** -24) / monthly
		assert abs(max_p - affordable * factor) < 1.0


class TestIFRS9:
	def test_ecl_stage1(self):
		ecl = ecl_stage1(100_000, 0.02, 0.40)
		assert ecl == pytest.approx(800.0, rel=0.01)

	def test_ecl_stage3(self):
		ecl = ecl_stage3(100_000, 0.40)
		assert ecl == pytest.approx(40_000.0, rel=0.01)

	def test_lgd_unsecured(self):
		assert lgd_from_collateral(False) == 0.40

	def test_lgd_well_secured(self):
		assert lgd_from_collateral(True, 1.5) == 0.15

	def test_pd_lifetime_extrapolation(self):
		lt = pd_lifetime(0.02, 60, 48)
		assert lt > 0.02
		assert lt <= 1.0


class TestCollateral:
	def test_fsv_property(self):
		fsv = forced_sale_value(1_000_000, "property")
		assert fsv == 600_000.0

	def test_fsv_vehicle(self):
		fsv = forced_sale_value(500_000, "vehicle")
		assert fsv == 350_000.0

	def test_fsv_cash(self):
		fsv = forced_sale_value(200_000, "cash")
		assert fsv == 190_000.0

	def test_collateral_coverage(self):
		ratio = collateral_coverage_ratio(150_000, 100_000)
		assert ratio == pytest.approx(1.5, rel=0.001)

	def test_coverage_zero_principal(self):
		assert collateral_coverage_ratio(100_000, 0) == float("inf")


class TestDPD:
	def test_dpd_bucket_current(self):
		assert dpd_bucket(0) == "current"

	def test_dpd_bucket_boundaries(self):
		assert dpd_bucket(1) == "1-30"
		assert dpd_bucket(30) == "1-30"
		assert dpd_bucket(31) == "31-60"
		assert dpd_bucket(90) == "61-90"
		assert dpd_bucket(91) == "91-120"
		assert dpd_bucket(121) == "120+"

	def test_ifrs9_stage(self):
		assert ifrs9_stage(0) == "stage1"
		assert ifrs9_stage(45) == "stage2"
		assert ifrs9_stage(91) == "stage3"

	def test_par_ratio(self):
		assert par_ratio(30_000, 100_000) == pytest.approx(0.30, rel=0.001)
		assert par_ratio(0, 100_000) == 0.0
		assert par_ratio(100_000, 0) == 0.0


class TestRatePricing:
	def test_risk_adjusted_rate(self):
		rate_a = risk_adjusted_rate(0.18, "A")
		rate_f = risk_adjusted_rate(0.18, "F")
		assert rate_f > rate_a
		assert rate_a == pytest.approx(0.18, rel=0.001)

	def test_grade_amount_cap(self):
		cap_a = grade_amount_cap(1_000_000, "A")
		cap_f = grade_amount_cap(1_000_000, "F")
		assert cap_a == 1_000_000.0
		assert cap_f == 200_000.0


class TestPenalties:
	def test_late_payment_penalty(self):
		penalty = late_payment_penalty(10_000, 0.02, 30)
		assert penalty > 0
		# Daily rate = 0.02/365; 30 days
		daily_r = 0.02 / 365
		expected = 10_000 * ((1 + daily_r) ** 30 - 1)
		assert penalty == pytest.approx(expected, rel=0.01)

	def test_zero_days_no_penalty(self):
		assert late_payment_penalty(10_000, 0.02, 0) == 0.0

	def test_processing_fee(self):
		fee = processing_fee(100_000, 0.015)
		assert fee == pytest.approx(1_500.0, rel=0.001)


class TestPortfolioAtRisk:
	def test_all_current(self):
		loans = [
			{
				"status": "active",
				"outstanding_principal": 100_000,
				"installments": [{"due_date": "2099-01-01", "status": "pending"}],
			}
		]
		assert portfolio_at_risk(loans, date.today()) == 0.0

	def test_all_delinquent(self):
		loans = [
			{
				"status": "active",
				"outstanding_principal": 100_000,
				"installments": [{"due_date": "2020-01-01", "status": "pending"}],
			}
		]
		assert portfolio_at_risk(loans, date.today(), dpd_threshold=30) == 1.0


class TestStressScenario:
	def test_stress_output(self):
		result = stress_scenario(1_000_000, 0.05, 0.10, lgd=0.40)
		assert result["incremental_loss"] == pytest.approx(40_000.0, rel=0.01)
		assert result["stressed_npl_ratio"] == pytest.approx(0.15, rel=0.001)

	def test_invalid_lgd(self):
		with pytest.raises(ValueError):
			stress_scenario(1_000_000, 0.05, 0.10, lgd=1.5)


# ---------------------------------------------------------------------------
# domain/rules.py
# ---------------------------------------------------------------------------

from domain.rules import (
	RuleViolation,
	assert_tenant_context, assert_actor_present, assert_no_cross_tenant,
	assert_valid_currency, assert_valid_product_type, assert_valid_rate,
	assert_valid_amount_limits, assert_valid_tenor,
	assert_kyc_present, assert_valid_purpose,
	assert_amount_within_product_limits, assert_tenor_within_product_limits,
	assert_no_duplicate_application, assert_borrower_not_blacklisted,
	assert_minimum_credit_score, assert_income_verified, assert_dsr_passes,
	assert_no_active_defaults, assert_no_fraud_flags,
	assert_valid_underwriting_decision, assert_decline_has_reason,
	assert_human_approval_for_high_value,
	assert_application_approved_for_offer, assert_offer_not_expired,
	assert_offer_accepted_for_disbursement,
	assert_valid_disbursement_rail, assert_disbursement_account_present,
	assert_loan_active_for_repayment,
	assert_dpd_for_demand_notice,
	calculate_required_provision_rate,
	assert_loan_eligible_for_restructure, assert_restructure_approved,
	assert_eligible_for_writeoff,
	assert_collateral_coverage,
	assert_single_obligor_limit,
	calculate_offer_tiers,
)


class TestTenantRules:
	def test_empty_tenant_raises(self):
		with pytest.raises(RuleViolation, match="tenant_context_required"):
			assert_tenant_context("")

	def test_valid_tenant_passes(self):
		assert_tenant_context("my_tenant")  # no exception

	def test_cross_tenant(self):
		with pytest.raises(RuleViolation, match="cross_tenant"):
			assert_no_cross_tenant("t1", "t2")

	def test_same_tenant_passes(self):
		assert_no_cross_tenant("t1", "t1")


class TestProductRules:
	def test_unsupported_currency(self):
		with pytest.raises(RuleViolation, match="unsupported_currency"):
			assert_valid_currency("XYZ")

	def test_supported_currency(self):
		assert_valid_currency("KES")
		assert_valid_currency("USD")

	def test_rate_too_high(self):
		with pytest.raises(RuleViolation):
			assert_valid_rate(0.80)

	def test_rate_too_low(self):
		with pytest.raises(RuleViolation):
			assert_valid_rate(0.001)

	def test_valid_rate(self):
		assert_valid_rate(0.18)

	def test_amount_limits_invalid(self):
		with pytest.raises(RuleViolation):
			assert_valid_amount_limits(500_000, 100_000)

	def test_tenor_invalid(self):
		with pytest.raises(RuleViolation):
			assert_valid_tenor(24, 12)


class TestApplicationRules:
	def test_kyc_required(self):
		with pytest.raises(RuleViolation, match="kyc_required"):
			assert_kyc_present("")

	def test_invalid_purpose(self):
		with pytest.raises(RuleViolation):
			assert_valid_purpose("gambling")

	def test_amount_below_min(self):
		with pytest.raises(RuleViolation, match="amount_below_minimum"):
			assert_amount_within_product_limits(500, 1_000, 50_000)

	def test_amount_above_max(self):
		with pytest.raises(RuleViolation, match="amount_above_maximum"):
			assert_amount_within_product_limits(60_000, 1_000, 50_000)

	def test_duplicate_application(self):
		with pytest.raises(RuleViolation, match="duplicate_application"):
			assert_no_duplicate_application(["submitted"])

	def test_blacklisted_borrower(self):
		with pytest.raises(RuleViolation, match="borrower_blacklisted"):
			assert_borrower_not_blacklisted(True, "B001")


class TestCreditRules:
	def test_score_below_minimum(self):
		with pytest.raises(RuleViolation, match="credit_score_below_minimum"):
			assert_minimum_credit_score(300)

	def test_income_not_verified(self):
		with pytest.raises(RuleViolation, match="income_not_verified"):
			assert_income_verified(False)

	def test_dsr_exceeds_threshold(self):
		with pytest.raises(RuleViolation, match="dsr_exceeds_threshold"):
			assert_dsr_passes(0.55)

	def test_active_defaults(self):
		with pytest.raises(RuleViolation, match="active_defaults"):
			assert_no_active_defaults(2)

	def test_fraud_flags(self):
		with pytest.raises(RuleViolation, match="fraud_flags"):
			assert_no_fraud_flags(["identity_fraud"])


class TestUnderwritingRules:
	def test_invalid_decision(self):
		with pytest.raises(RuleViolation):
			assert_valid_underwriting_decision("maybe")

	def test_decline_needs_reason(self):
		with pytest.raises(RuleViolation, match="decline_requires_adverse_reason"):
			assert_decline_has_reason("decline", "")

	def test_high_value_needs_approval(self):
		with pytest.raises(RuleViolation, match="high_value_requires_human_approval"):
			assert_human_approval_for_high_value(600_000, "")


class TestOfferRules:
	def test_offer_expired(self):
		with pytest.raises(RuleViolation, match="offer_expired"):
			assert_offer_not_expired(date(2020, 1, 1))

	def test_offer_not_accepted(self):
		with pytest.raises(RuleViolation, match="offer_not_accepted"):
			assert_offer_accepted_for_disbursement("issued")


class TestCollectionRules:
	def test_demand_notice_dpd_insufficient(self):
		with pytest.raises(RuleViolation, match="dpd_insufficient_for_notice_level"):
			assert_dpd_for_demand_notice(0, 1)

	def test_provision_rate_current(self):
		rate = calculate_required_provision_rate(0, False)
		assert rate == 0.01

	def test_provision_rate_stage3(self):
		rate = calculate_required_provision_rate(200, False)
		assert rate == 1.00


class TestRestructureRules:
	def test_max_restructures(self):
		with pytest.raises(RuleViolation, match="max_restructures_reached"):
			assert_loan_eligible_for_restructure("active", 3)

	def test_loan_not_active(self):
		with pytest.raises(RuleViolation):
			assert_loan_eligible_for_restructure("closed", 0)


class TestWriteOffRules:
	def test_insufficient_dpd(self):
		with pytest.raises(RuleViolation, match="insufficient_dpd"):
			assert_eligible_for_writeoff(60, "manager")

	def test_no_approver(self):
		with pytest.raises(RuleViolation, match="writeoff_requires_approval"):
			assert_eligible_for_writeoff(180, "")


class TestOfferTierCalc:
	def test_three_tiers_for_grade_a(self):
		tiers = calculate_offer_tiers(100_000, 0.18, 24, "A")
		assert len(tiers) == 3
		names = {t["tier"] for t in tiers}
		assert "aggressive" in names

	def test_two_tiers_for_grade_d(self):
		tiers = calculate_offer_tiers(100_000, 0.18, 24, "D")
		assert len(tiers) == 2
		names = {t["tier"] for t in tiers}
		assert "aggressive" not in names

	def test_tier_amounts_ascending(self):
		tiers = calculate_offer_tiers(100_000, 0.18, 24, "B")
		amounts = [t["offered_amount"] for t in tiers]
		assert amounts == sorted(amounts)
