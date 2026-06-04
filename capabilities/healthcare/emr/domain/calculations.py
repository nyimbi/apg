"""Clinical domain calculations for Electronic Medical Records.

All formulas are pure functions with typed inputs and edge-case handling.
No I/O, no side effects.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any


# ── BMI ────────────────────────────────────────────────────────────────────────

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
	"""BMI = weight(kg) / height(m)²."""
	if height_cm <= 0 or weight_kg <= 0:
		raise ValueError("weight and height must be positive")
	height_m = height_cm / 100.0
	return round(weight_kg / (height_m ** 2), 1)


def bmi_category(bmi: float) -> str:
	"""WHO adult BMI classification."""
	if bmi < 18.5:
		return "underweight"
	if bmi < 25.0:
		return "normal"
	if bmi < 30.0:
		return "overweight"
	if bmi < 35.0:
		return "obese_class_i"
	if bmi < 40.0:
		return "obese_class_ii"
	return "obese_class_iii"


# ── paediatric dosing ──────────────────────────────────────────────────────────

def calculate_paediatric_dose_mg(dose_per_kg: float, weight_kg: float, max_dose_mg: float | None = None) -> float:
	"""Weight-based paediatric dose; optionally capped at adult max."""
	if dose_per_kg <= 0 or weight_kg <= 0:
		raise ValueError("dose_per_kg and weight_kg must be positive")
	dose = round(dose_per_kg * weight_kg, 2)
	if max_dose_mg is not None:
		dose = min(dose, max_dose_mg)
	return dose


def estimate_weight_by_age_kg(age_months: int) -> float:
	"""Luscombe–Owens formula for children 1–12 years when weight unavailable."""
	if age_months < 1:
		return 3.5   # average neonate weight
	if age_months <= 12:
		# months → kg: ~0.5 kg/month gain up to 1 year
		return round(3.5 + age_months * 0.5, 1)
	age_years = age_months / 12.0
	if age_years <= 12:
		return round(2 * age_years + 8, 1)   # Broselow approximation
	return round(3 * age_years + 7, 1)        # adolescent approximation


def clark_rule_dose(adult_dose_mg: float, weight_kg: float, average_adult_weight_kg: float = 68.0) -> float:
	"""Clark's rule: child dose = (weight_kg / avg_adult_weight) × adult_dose."""
	if weight_kg <= 0 or average_adult_weight_kg <= 0:
		raise ValueError("weights must be positive")
	return round((weight_kg / average_adult_weight_kg) * adult_dose_mg, 2)


# ── renal dosing ───────────────────────────────────────────────────────────────

def cockroft_gault_egfr(
	age_years: int,
	weight_kg: float,
	serum_creatinine_umol_L: float,
	is_female: bool,
) -> float:
	"""Cockcroft-Gault estimated GFR in mL/min.

	CG: ((140 - age) × weight) / (72 × Scr_mg/dL) × 0.85 if female
	Converts µmol/L → mg/dL internally.
	"""
	if serum_creatinine_umol_L <= 0 or weight_kg <= 0 or age_years <= 0:
		raise ValueError("all inputs must be positive")
	scr_mgdl = serum_creatinine_umol_L / 88.42
	egfr = ((140 - age_years) * weight_kg) / (72 * scr_mgdl)
	if is_female:
		egfr *= 0.85
	return round(egfr, 1)


def renal_dose_adjustment_factor(egfr: float) -> float:
	"""Return a dose reduction factor (0–1.0) based on eGFR stage."""
	if egfr >= 60:
		return 1.0
	if egfr >= 30:
		return 0.75
	if egfr >= 15:
		return 0.50
	return 0.25   # severe CKD / ESRD — specialist review required


def ckd_stage(egfr: float) -> str:
	"""KDIGO CKD staging."""
	if egfr >= 90:
		return "G1_normal_or_high"
	if egfr >= 60:
		return "G2_mildly_decreased"
	if egfr >= 45:
		return "G3a_mild_moderate"
	if egfr >= 30:
		return "G3b_moderate_severe"
	if egfr >= 15:
		return "G4_severely_decreased"
	return "G5_kidney_failure"


# ── hepatic dosing ─────────────────────────────────────────────────────────────

def child_pugh_score(
	bilirubin_umol_L: float,
	albumin_g_L: float,
	inr: float,
	ascites: str,       # "none" | "mild" | "moderate_severe"
	encephalopathy: str,  # "none" | "grade_1_2" | "grade_3_4"
) -> tuple[int, str]:
	"""Child-Pugh score for hepatic function classification.

	Returns (score, class) where class is A (5-6), B (7-9), or C (10-15).
	"""
	score = 0

	# bilirubin (µmol/L): <34 → 1, 34–51 → 2, >51 → 3
	if bilirubin_umol_L < 34:
		score += 1
	elif bilirubin_umol_L <= 51:
		score += 2
	else:
		score += 3

	# albumin (g/L): >35 → 1, 28–35 → 2, <28 → 3
	if albumin_g_L > 35:
		score += 1
	elif albumin_g_L >= 28:
		score += 2
	else:
		score += 3

	# INR: <1.7 → 1, 1.7–2.2 → 2, >2.2 → 3
	if inr < 1.7:
		score += 1
	elif inr <= 2.2:
		score += 2
	else:
		score += 3

	# ascites
	score += {"none": 1, "mild": 2, "moderate_severe": 3}.get(ascites, 1)

	# encephalopathy
	score += {"none": 1, "grade_1_2": 2, "grade_3_4": 3}.get(encephalopathy, 1)

	if score <= 6:
		return score, "A"
	if score <= 9:
		return score, "B"
	return score, "C"


def hepatic_dose_adjustment_factor(child_pugh_class: str) -> float:
	"""Conservative dose reduction for hepatic impairment."""
	return {"A": 1.0, "B": 0.75, "C": 0.50}.get(child_pugh_class, 1.0)


# ── vital sign interpretation ──────────────────────────────────────────────────

def interpret_blood_pressure(systolic: float, diastolic: float) -> str:
	"""ACC/AHA 2017 hypertension classification."""
	if systolic < 120 and diastolic < 80:
		return "normal"
	if systolic < 130 and diastolic < 80:
		return "elevated"
	if systolic < 140 or diastolic < 90:
		return "hypertension_stage_1"
	if systolic < 180 or diastolic < 120:
		return "hypertension_stage_2"
	return "hypertensive_crisis"


def interpret_oxygen_saturation(spo2: float) -> str:
	if spo2 >= 95:
		return "normal"
	if spo2 >= 90:
		return "mild_hypoxaemia"
	if spo2 >= 85:
		return "moderate_hypoxaemia"
	return "severe_hypoxaemia"


def interpret_temperature_celsius(temp_c: float) -> str:
	if temp_c < 35.0:
		return "hypothermia"
	if temp_c < 37.2:
		return "normal"
	if temp_c < 38.3:
		return "low_grade_fever"
	if temp_c < 39.4:
		return "moderate_fever"
	if temp_c < 41.0:
		return "high_fever"
	return "hyperpyrexia"


def is_critical_vital(vital_type: str, value: float, value2: float | None = None) -> bool:
	"""Return True if a vital sign value is in a clinically critical range."""
	checks: dict[str, Any] = {
		"heart_rate": lambda v, _: v < 40 or v > 150,
		"oxygen_saturation": lambda v, _: v < 90,
		"temperature": lambda v, _: v < 35.0 or v >= 41.0,
		"blood_pressure": lambda v, d: v > 180 or (d is not None and d > 120) or v < 70,
		"blood_glucose": lambda v, _: v < 2.8 or v > 22.2,
	}
	fn = checks.get(vital_type)
	if fn is None:
		return False
	return fn(value, value2)


# ── age / date helpers ─────────────────────────────────────────────────────────

def age_in_years(birth_date: date, reference: date | None = None) -> int:
	ref = reference or date.today()
	return ref.year - birth_date.year - (
		(ref.month, ref.day) < (birth_date.month, birth_date.day)
	)


def age_in_months(birth_date: date, reference: date | None = None) -> int:
	ref = reference or date.today()
	years = ref.year - birth_date.year
	months = ref.month - birth_date.month
	if ref.day < birth_date.day:
		months -= 1
	return years * 12 + months


# ── probabilistic patient matching ────────────────────────────────────────────

def _soundex(name: str) -> str:
	"""Simple Soundex — not production-grade but sufficient as a component."""
	if not name:
		return ""
	name = name.upper()
	codes = {"BFPV": "1", "CGJKQSXYZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}
	result = name[0]
	prev = "0"
	for ch in name[1:]:
		code = "0"
		for letters, c in codes.items():
			if ch in letters:
				code = c
				break
		if code != "0" and code != prev:
			result += code
		prev = code
	return (result + "000")[:4]


def patient_match_score(
	a: dict[str, Any],
	b: dict[str, Any],
) -> tuple[float, list[str]]:
	"""Probabilistic patient matching using weighted field comparison.

	Inputs are dicts with keys: family, given_0, birth_date, gender,
	national_id, phone, biometric_hash.
	Returns (score 0-1, list of matching field names).
	"""
	score = 0.0
	matched: list[str] = []

	# biometric hash — near-certain match
	if a.get("biometric_hash") and a.get("biometric_hash") == b.get("biometric_hash"):
		score += 0.60
		matched.append("biometric_hash")

	# national ID — strong match
	if a.get("national_id") and a.get("national_id") == b.get("national_id"):
		score += 0.35
		matched.append("national_id")

	# birth date
	if a.get("birth_date") and a.get("birth_date") == b.get("birth_date"):
		score += 0.15
		matched.append("birth_date")

	# family name (exact + soundex)
	a_fam = (a.get("family") or "").upper()
	b_fam = (b.get("family") or "").upper()
	if a_fam and a_fam == b_fam:
		score += 0.15
		matched.append("family_name_exact")
	elif _soundex(a_fam) == _soundex(b_fam) and a_fam and b_fam:
		score += 0.07
		matched.append("family_name_soundex")

	# given name
	a_giv = (a.get("given_0") or "").upper()
	b_giv = (b.get("given_0") or "").upper()
	if a_giv and a_giv == b_giv:
		score += 0.10
		matched.append("given_name_exact")

	# gender
	if a.get("gender") and a.get("gender") == b.get("gender"):
		score += 0.05
		matched.append("gender")

	# phone
	if a.get("phone") and a.get("phone") == b.get("phone"):
		score += 0.10
		matched.append("phone")

	return round(min(score, 1.0), 3), matched
