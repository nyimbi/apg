"""Tests for domain calculations — Patient Management."""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from pmt.domain.calculations import (
	calculate_age_months,
	calculate_age_years,
	calculate_avg_los,
	calculate_bed_turnover_rate,
	calculate_bill_balance_due,
	calculate_bill_subtotal,
	calculate_bill_tax,
	calculate_bill_total,
	calculate_collection_rate,
	calculate_composite_satisfaction,
	calculate_days_in_ar,
	calculate_denial_rate,
	calculate_early_warning_score,
	calculate_installment_amount,
	calculate_los_days,
	calculate_los_hours,
	calculate_nhif_benefit,
	calculate_no_show_risk,
	calculate_nps_bucket,
	calculate_occupancy_rate,
	calculate_patient_responsibility,
	calculate_readmission_risk_score,
	calculate_waitlist_priority_score,
	effective_available_beds,
	generate_mrn,
	is_deposit_adequate,
	is_overflow_risk,
	is_paediatric,
)


# ── MRN ───────────────────────────────────────────────────────────────────────

def test_generate_mrn_format():
	mrn = generate_mrn("nairobi", 1)
	assert mrn == "MRNNAIR00000001"


def test_generate_mrn_pads_short_prefix():
	mrn = generate_mrn("ab", 5)
	assert mrn.startswith("MRNABXX")


# ── age ────────────────────────────────────────────────────────────────────────

def test_calculate_age_years_basic():
	dob = datetime(1985, 6, 15)
	ref = datetime(2026, 6, 15)
	assert calculate_age_years(dob, ref) == 41


def test_calculate_age_years_before_birthday():
	dob = datetime(1985, 6, 15)
	ref = datetime(2026, 6, 14)
	assert calculate_age_years(dob, ref) == 40


def test_calculate_age_months():
	dob = datetime(2024, 1, 1)
	ref = datetime(2026, 3, 1)
	assert calculate_age_months(dob, ref) == 26


def test_is_paediatric_true():
	dob = datetime(2015, 1, 1)
	assert is_paediatric(dob) is True


def test_is_paediatric_false():
	dob = datetime(1990, 1, 1)
	assert is_paediatric(dob) is False


# ── LOS ───────────────────────────────────────────────────────────────────────

def test_calculate_los_hours():
	admit = datetime(2026, 1, 1, 8, 0)
	discharge = datetime(2026, 1, 3, 8, 0)
	assert calculate_los_hours(admit, discharge) == 48.0


def test_calculate_los_days():
	admit = datetime(2026, 1, 1, 0, 0)
	discharge = datetime(2026, 1, 4, 0, 0)
	assert calculate_los_days(admit, discharge) == 3.0


def test_calculate_avg_los_empty():
	assert calculate_avg_los([]) == 0.0


def test_calculate_avg_los_values():
	assert calculate_avg_los([24.0, 48.0, 72.0]) == 48.0


# ── bed occupancy ─────────────────────────────────────────────────────────────

def test_occupancy_rate_zero_total():
	assert calculate_occupancy_rate(0, 0) == 0.0


def test_occupancy_rate():
	assert calculate_occupancy_rate(8, 10) == 80.0


def test_overflow_risk_true():
	assert is_overflow_risk(0, 20) is True


def test_overflow_risk_false():
	assert is_overflow_risk(5, 20) is False


def test_effective_available_beds():
	result = effective_available_beds(available=3, cleaning=4)
	assert result >= 3


def test_bed_turnover_rate():
	rate = calculate_bed_turnover_rate(discharges=60, total_beds=20, period_days=30)
	assert rate == pytest.approx(0.1, abs=0.0001)


# ── waitlist priority ─────────────────────────────────────────────────────────

def test_emergency_priority_highest():
	score_emerg = calculate_waitlist_priority_score("emergency", 0)
	score_routine = calculate_waitlist_priority_score("routine", 0)
	assert score_emerg > score_routine


def test_isolation_modifier():
	base = calculate_waitlist_priority_score("urgent", 0, isolation_required=False)
	with_iso = calculate_waitlist_priority_score("urgent", 0, isolation_required=True)
	assert with_iso - base == pytest.approx(5.0)


def test_paediatric_modifier():
	base = calculate_waitlist_priority_score("routine", 0, paediatric=False)
	paed = calculate_waitlist_priority_score("routine", 0, paediatric=True)
	assert paed - base == pytest.approx(3.0)


# ── bill calculations ─────────────────────────────────────────────────────────

def test_bill_subtotal():
	prices = [Decimal("500"), Decimal("1000")]
	qtys = [2, 1]
	assert calculate_bill_subtotal(prices, qtys) == Decimal("2000.00")


def test_bill_subtotal_length_mismatch():
	with pytest.raises(ValueError):
		calculate_bill_subtotal([Decimal("100")], [1, 2])


def test_bill_tax_16pct():
	assert calculate_bill_tax(Decimal("1000")) == Decimal("160.00")


def test_bill_total():
	assert calculate_bill_total(Decimal("1000"), Decimal("160")) == Decimal("1160.00")


def test_bill_balance_floor():
	bal = calculate_bill_balance_due(
		subtotal=Decimal("1000"),
		insurance_adjustment=Decimal("600"),
		write_off_amount=Decimal("600"),  # intentional over-adjustment
		amount_paid=Decimal("0"),
	)
	assert bal == Decimal("0.00")


def test_patient_responsibility_capped_by_oop_max():
	resp = calculate_patient_responsibility(
		total_billed=Decimal("100000"),
		adjudicated_amount=Decimal("80000"),
		copay=Decimal("500"),
		deductible=Decimal("2000"),
		coinsurance_pct=Decimal("0.20"),
		out_of_pocket_max=Decimal("5000"),
	)
	assert resp <= Decimal("5000")


def test_patient_responsibility_coinsurance_invalid():
	with pytest.raises(ValueError):
		calculate_patient_responsibility(
			total_billed=Decimal("1000"),
			adjudicated_amount=Decimal("800"),
			coinsurance_pct=Decimal("1.5"),
		)


def test_installment_amount():
	result = calculate_installment_amount(Decimal("12000"), 6)
	assert result == Decimal("2000.00")


# ── collection rate ───────────────────────────────────────────────────────────

def test_collection_rate():
	assert calculate_collection_rate(Decimal("100"), Decimal("95")) == 95.0


def test_collection_rate_zero():
	assert calculate_collection_rate(Decimal("0"), Decimal("0")) == 0.0


def test_denial_rate():
	assert calculate_denial_rate(100, 5) == 5.0


def test_days_in_ar():
	result = calculate_days_in_ar(Decimal("30000"), Decimal("1000"))
	assert result == 30.0


# ── no-show risk ──────────────────────────────────────────────────────────────

def test_no_show_risk_telehealth_lower():
	in_person = calculate_no_show_risk(2, 1, 10, 7, False)
	tele = calculate_no_show_risk(2, 1, 10, 7, True)
	assert tele < in_person


def test_no_show_risk_bounds():
	score = calculate_no_show_risk(0, 0, 0, 0, False)
	assert 0.0 <= score <= 1.0


# ── readmission risk ──────────────────────────────────────────────────────────

def test_readmission_risk_high_age():
	r1 = calculate_readmission_risk_score(0, 30, False, True, True)
	r2 = calculate_readmission_risk_score(0, 80, False, True, True)
	assert r2 > r1


def test_readmission_risk_no_discharge_plan():
	r1 = calculate_readmission_risk_score(0, 50, False, True, True)
	r2 = calculate_readmission_risk_score(0, 50, False, False, True)
	assert r2 > r1


def test_readmission_risk_capped():
	score = calculate_readmission_risk_score(5, 90, True, False, False)
	assert score <= 1.0


# ── early warning score ───────────────────────────────────────────────────────

def test_ews_normal_vitals():
	score, level = calculate_early_warning_score({
		"bp_systolic": 120, "respiratory_rate": 16, "spo2": 98,
		"heart_rate": 75, "temperature_c": 37.0, "avpu_score": 1.0,
	})
	assert score == 0 and level == "low"


def test_ews_critical():
	score, level = calculate_early_warning_score({
		"bp_systolic": 80, "respiratory_rate": 30, "spo2": 88,
		"heart_rate": 140, "temperature_c": 38.5, "avpu_score": 0.0,
	})
	assert level in ("high", "critical")


# ── satisfaction ──────────────────────────────────────────────────────────────

def test_nps_promoter():
	assert calculate_nps_bucket(9.5) == "promoter"


def test_nps_passive():
	assert calculate_nps_bucket(8.0) == "passive"


def test_nps_detractor():
	assert calculate_nps_bucket(5.0) == "detractor"


def test_composite_satisfaction_excludes_nps():
	result = calculate_composite_satisfaction({"overall": 4, "wait_time": 3, "would_recommend": 10})
	assert result == pytest.approx(3.5)


def test_composite_satisfaction_empty():
	assert calculate_composite_satisfaction({}) is None


# ── NHIF benefit ──────────────────────────────────────────────────────────────

def test_nhif_general_emergency():
	benefit = calculate_nhif_benefit("emergency", 3, "general")
	assert benefit == Decimal("7500.00")


def test_nhif_icu():
	benefit = calculate_nhif_benefit("elective", 2, "icu")
	assert benefit == Decimal("16000.00")


def test_nhif_min_1_day():
	benefit = calculate_nhif_benefit("elective", 0, "general")
	assert benefit >= Decimal("1800.00")


# ── deposit adequacy ──────────────────────────────────────────────────────────

def test_deposit_adequate():
	assert is_deposit_adequate(Decimal("3000"), Decimal("10000"), 30.0) is True


def test_deposit_inadequate():
	assert is_deposit_adequate(Decimal("100"), Decimal("10000"), 30.0) is False


def test_deposit_zero_bill():
	assert is_deposit_adequate(Decimal("0"), Decimal("0")) is True
