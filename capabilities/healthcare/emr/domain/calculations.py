"""Clinical domain calculations for Electronic Medical Records.

All functions are pure: no I/O, no side-effects.
Inputs are typed and edge cases are handled with ValueError.
"""
from __future__ import annotations

import math
from datetime import date, datetime
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

def calculate_paediatric_dose_mg(
	dose_per_kg: float,
	weight_kg: float,
	max_dose_mg: float | None = None,
) -> float:
	"""Weight-based paediatric dose, optionally capped at adult maximum."""
	if dose_per_kg <= 0 or weight_kg <= 0:
		raise ValueError("dose_per_kg and weight_kg must be positive")
	dose = round(dose_per_kg * weight_kg, 2)
	if max_dose_mg is not None:
		dose = min(dose, max_dose_mg)
	return dose


def estimate_weight_by_age_kg(age_months: int) -> float:
	"""Luscombe–Owens / Broselow estimate when actual weight is unavailable."""
	if age_months < 1:
		return 3.5   # average neonate
	if age_months <= 12:
		return round(3.5 + age_months * 0.5, 1)
	age_years = age_months / 12.0
	if age_years <= 12:
		return round(2 * age_years + 8, 1)
	return round(3 * age_years + 7, 1)


def clark_rule_dose(
	adult_dose_mg: float,
	weight_kg: float,
	average_adult_weight_kg: float = 68.0,
) -> float:
	"""Clark's rule: child dose = (weight_kg / avg_adult_weight) × adult_dose."""
	if weight_kg <= 0 or average_adult_weight_kg <= 0:
		raise ValueError("weights must be positive")
	return round((weight_kg / average_adult_weight_kg) * adult_dose_mg, 2)


def young_rule_dose(adult_dose_mg: float, age_years: float) -> float:
	"""Young's rule: child dose = (age / (age + 12)) × adult_dose.
	Valid for children 2–12 years."""
	if age_years <= 0:
		raise ValueError("age_years must be positive")
	return round((age_years / (age_years + 12)) * adult_dose_mg, 2)


def fried_rule_dose_infant(adult_dose_mg: float, age_months: int) -> float:
	"""Fried's rule for infants <2 years: dose = (age_months / 150) × adult_dose."""
	if age_months < 0:
		raise ValueError("age_months must be non-negative")
	return round((age_months / 150.0) * adult_dose_mg, 2)


# ── renal dosing ───────────────────────────────────────────────────────────────

def cockroft_gault_egfr(
	age_years: int,
	weight_kg: float,
	serum_creatinine_umol_L: float,
	is_female: bool,
) -> float:
	"""Cockcroft-Gault eGFR in mL/min. Converts µmol/L → mg/dL internally."""
	if serum_creatinine_umol_L <= 0 or weight_kg <= 0 or age_years <= 0:
		raise ValueError("all inputs must be positive")
	scr_mgdl = serum_creatinine_umol_L / 88.42
	egfr = ((140 - age_years) * weight_kg) / (72 * scr_mgdl)
	if is_female:
		egfr *= 0.85
	return round(egfr, 1)


def mdrd_egfr(
	serum_creatinine_mgdl: float,
	age_years: int,
	is_female: bool,
	is_african_american: bool = False,
) -> float:
	"""MDRD 4-variable eGFR equation (mL/min/1.73 m²).

	186 × Scr^-1.154 × Age^-0.203 × 0.742 (female) × 1.212 (AA).
	"""
	if serum_creatinine_mgdl <= 0 or age_years <= 0:
		raise ValueError("creatinine and age must be positive")
	egfr = 186 * (serum_creatinine_mgdl ** -1.154) * (age_years ** -0.203)
	if is_female:
		egfr *= 0.742
	if is_african_american:
		egfr *= 1.212
	return round(egfr, 1)


def renal_dose_adjustment_factor(egfr: float) -> float:
	"""Conservative dose reduction factor (0–1.0) based on eGFR stage."""
	if egfr >= 60:
		return 1.0
	if egfr >= 30:
		return 0.75
	if egfr >= 15:
		return 0.50
	return 0.25   # ESRD — specialist review required


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
	ascites: str,          # "none" | "mild" | "moderate_severe"
	encephalopathy: str,   # "none" | "grade_1_2" | "grade_3_4"
) -> tuple[int, str]:
	"""Child-Pugh hepatic function score → (score, class A/B/C)."""
	score = 0
	if bilirubin_umol_L < 34:
		score += 1
	elif bilirubin_umol_L <= 51:
		score += 2
	else:
		score += 3
	if albumin_g_L > 35:
		score += 1
	elif albumin_g_L >= 28:
		score += 2
	else:
		score += 3
	if inr < 1.7:
		score += 1
	elif inr <= 2.2:
		score += 2
	else:
		score += 3
	score += {"none": 1, "mild": 2, "moderate_severe": 3}.get(ascites, 1)
	score += {"none": 1, "grade_1_2": 2, "grade_3_4": 3}.get(encephalopathy, 1)
	cls = "A" if score <= 6 else "B" if score <= 9 else "C"
	return score, cls


def hepatic_dose_adjustment_factor(child_pugh_class: str) -> float:
	"""Conservative dose reduction for hepatic impairment (Child-Pugh A/B/C)."""
	return {"A": 1.0, "B": 0.75, "C": 0.50}.get(child_pugh_class, 1.0)


def meld_score(
	inr: float,
	bilirubin_mgdl: float,
	creatinine_mgdl: float,
) -> int:
	"""MELD score for liver disease severity (used in transplant allocation).

	MELD = 3.78 × ln(bilirubin) + 11.2 × ln(INR) + 9.57 × ln(creatinine) + 6.43
	Clamp bilirubin/creatinine at min 1.0. Creatinine capped at 4.0.
	"""
	bili = max(bilirubin_mgdl, 1.0)
	cr = min(max(creatinine_mgdl, 1.0), 4.0)
	inr_val = max(inr, 1.0)
	raw = 3.78 * math.log(bili) + 11.2 * math.log(inr_val) + 9.57 * math.log(cr) + 6.43
	return max(6, min(40, round(raw)))


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


def interpret_heart_rate(hr: float, age_years: int = 30) -> str:
	"""Normal adult: 60–100 bpm. Paediatric ranges differ."""
	if age_years < 1:
		low, high = 100, 160
	elif age_years < 5:
		low, high = 80, 120
	elif age_years < 12:
		low, high = 70, 110
	else:
		low, high = 60, 100
	if hr < low:
		return "bradycardia"
	if hr > high:
		return "tachycardia"
	return "normal"


def interpret_respiratory_rate(rr: float, age_years: int = 30) -> str:
	if age_years < 1:
		low, high = 30, 60
	elif age_years < 5:
		low, high = 22, 40
	elif age_years < 12:
		low, high = 18, 30
	else:
		low, high = 12, 20
	if rr < low:
		return "bradypnoea"
	if rr > high:
		return "tachypnoea"
	return "normal"


def is_critical_vital(vital_type: str, value: float, value2: float | None = None) -> bool:
	"""Return True if a vital sign value is in a clinically critical range."""
	checks: dict[str, Any] = {
		"heart_rate": lambda v, _: v < 40 or v > 150,
		"oxygen_saturation": lambda v, _: v < 90,
		"temperature": lambda v, _: v < 35.0 or v >= 41.0,
		"blood_pressure": lambda v, d: v > 180 or (d is not None and d > 120) or v < 70,
		"blood_glucose": lambda v, _: v < 2.8 or v > 22.2,
		"respiratory_rate": lambda v, _: v < 8 or v > 30,
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


def gestational_age_weeks(lmp_date: date, reference: date | None = None) -> float:
	"""Gestational age in weeks from last menstrual period."""
	ref = reference or date.today()
	days = (ref - lmp_date).days
	return round(days / 7.0, 1)


def estimated_due_date(lmp_date: date) -> date:
	"""Naegele's rule: EDD = LMP + 280 days."""
	from datetime import timedelta
	return lmp_date + timedelta(days=280)


# ── fluid / electrolyte ────────────────────────────────────────────────────────

def maintenance_fluid_rate_ml_per_hour(weight_kg: float) -> float:
	"""Holliday-Segar formula for daily maintenance fluids in mL/hour."""
	if weight_kg <= 0:
		raise ValueError("weight_kg must be positive")
	if weight_kg <= 10:
		daily_ml = weight_kg * 100
	elif weight_kg <= 20:
		daily_ml = 1000 + (weight_kg - 10) * 50
	else:
		daily_ml = 1500 + (weight_kg - 20) * 20
	return round(daily_ml / 24, 1)


def sodium_deficit_mmol(
	desired_na_mmol_L: float,
	actual_na_mmol_L: float,
	weight_kg: float,
	is_female: bool = False,
) -> float:
	"""Sodium deficit = TBW × (target_Na - actual_Na).
	TBW = 0.6 × weight (male) or 0.5 × weight (female).
	"""
	tbw = weight_kg * (0.5 if is_female else 0.6)
	return round(tbw * (desired_na_mmol_L - actual_na_mmol_L), 1)


def anion_gap(
	sodium_mmol_L: float,
	chloride_mmol_L: float,
	bicarbonate_mmol_L: float,
) -> float:
	"""Standard anion gap = Na – (Cl + HCO3). Normal: 8–12 mEq/L."""
	return round(sodium_mmol_L - (chloride_mmol_L + bicarbonate_mmol_L), 1)


# ── cardiovascular risk ────────────────────────────────────────────────────────

def framingham_10yr_cvd_risk(
	age_years: int,
	total_cholesterol_mmol_L: float,
	hdl_cholesterol_mmol_L: float,
	systolic_bp: float,
	on_bp_treatment: bool,
	smoker: bool,
	is_female: bool,
) -> float:
	"""Simplified Framingham 10-year CVD risk score (%).

	Uses the Wilson 1998 point-score approximation.
	Returns risk as a percentage (0–100).
	"""
	# convert mmol/L to mg/dL
	tc_mgdl = total_cholesterol_mmol_L * 38.67
	hdl_mgdl = hdl_cholesterol_mmol_L * 38.67

	if is_female:
		# age points
		if age_years < 30:
			pts = -9
		elif age_years < 35:
			pts = -4
		elif age_years < 40:
			pts = 0
		elif age_years < 45:
			pts = 3
		elif age_years < 50:
			pts = 6
		elif age_years < 55:
			pts = 7
		elif age_years < 60:
			pts = 8
		elif age_years < 65:
			pts = 8
		elif age_years < 70:
			pts = 8
		elif age_years < 75:
			pts = 8
		else:
			pts = 8
		# total cholesterol
		if tc_mgdl < 160:
			pts += -2
		elif tc_mgdl < 200:
			pts += 0
		elif tc_mgdl < 240:
			pts += 1
		elif tc_mgdl < 280:
			pts += 1
		else:
			pts += 3
		# HDL
		if hdl_mgdl >= 60:
			pts += -2
		elif hdl_mgdl >= 50:
			pts += 0
		elif hdl_mgdl >= 40:
			pts += 1
		else:
			pts += 2
		# SBP (treated / untreated)
		if on_bp_treatment:
			if systolic_bp < 120:
				pts += -1
			elif systolic_bp < 130:
				pts += 2
			elif systolic_bp < 140:
				pts += 3
			elif systolic_bp < 150:
				pts += 5
			elif systolic_bp < 160:
				pts += 6
			else:
				pts += 7
		else:
			if systolic_bp < 120:
				pts += -3
			elif systolic_bp < 130:
				pts += 0
			elif systolic_bp < 140:
				pts += 1
			elif systolic_bp < 150:
				pts += 2
			elif systolic_bp < 160:
				pts += 4
			else:
				pts += 5
		if smoker:
			pts += 2
		# risk table (approximate)
		risk_table = {-3: 1, -2: 1, -1: 2, 0: 2, 1: 2, 2: 3, 3: 3, 4: 4, 5: 4,
					  6: 5, 7: 6, 8: 7, 9: 8, 10: 10, 11: 11, 12: 13, 13: 15,
					  14: 18, 15: 20, 16: 24, 17: 27}
		return float(risk_table.get(pts, 30 if pts > 17 else 1))
	else:
		# male (simplified)
		if age_years < 35:
			pts = 0
		elif age_years < 40:
			pts = 2
		elif age_years < 45:
			pts = 5
		elif age_years < 50:
			pts = 6
		elif age_years < 55:
			pts = 8
		elif age_years < 60:
			pts = 10
		elif age_years < 65:
			pts = 11
		elif age_years < 70:
			pts = 12
		elif age_years < 75:
			pts = 14
		else:
			pts = 15
		if tc_mgdl < 160:
			pts += -3
		elif tc_mgdl < 200:
			pts += 0
		elif tc_mgdl < 240:
			pts += 1
		elif tc_mgdl < 280:
			pts += 2
		else:
			pts += 3
		if hdl_mgdl >= 60:
			pts += -2
		elif hdl_mgdl >= 50:
			pts += 0
		elif hdl_mgdl >= 40:
			pts += 1
		else:
			pts += 2
		if on_bp_treatment:
			if systolic_bp < 120:
				pts += 0
			elif systolic_bp < 130:
				pts += 2
			elif systolic_bp < 140:
				pts += 3
			elif systolic_bp < 160:
				pts += 4
			else:
				pts += 5
		else:
			if systolic_bp < 120:
				pts += -2
			elif systolic_bp < 130:
				pts += 0
			elif systolic_bp < 140:
				pts += 1
			elif systolic_bp < 160:
				pts += 2
			else:
				pts += 3
		if smoker:
			pts += 4
		risk_table_m = {-3: 1, -2: 1, -1: 2, 0: 2, 1: 2, 2: 2, 3: 3, 4: 4, 5: 4,
						6: 5, 7: 6, 8: 8, 9: 10, 10: 12, 11: 14, 12: 16, 13: 20,
						14: 22, 15: 27}
		return float(risk_table_m.get(pts, 30 if pts > 15 else 1))


# ── probabilistic patient matching ────────────────────────────────────────────

def _soundex(name: str) -> str:
	"""Simple Soundex implementation (4-character code)."""
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


def _jaro_winkler(s1: str, s2: str) -> float:
	"""Jaro-Winkler similarity between two strings (0–1)."""
	if s1 == s2:
		return 1.0
	if not s1 or not s2:
		return 0.0
	len1, len2 = len(s1), len(s2)
	match_distance = max(len1, len2) // 2 - 1
	s1_matches = [False] * len1
	s2_matches = [False] * len2
	matches = 0
	transpositions = 0
	for i in range(len1):
		start = max(0, i - match_distance)
		end = min(i + match_distance + 1, len2)
		for j in range(start, end):
			if s2_matches[j] or s1[i] != s2[j]:
				continue
			s1_matches[i] = s2_matches[j] = True
			matches += 1
			break
	if not matches:
		return 0.0
	k = 0
	for i in range(len1):
		if not s1_matches[i]:
			continue
		while not s2_matches[k]:
			k += 1
		if s1[i] != s2[k]:
			transpositions += 1
		k += 1
	jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
	prefix = 0
	for i in range(min(4, len1, len2)):
		if s1[i] == s2[i]:
			prefix += 1
		else:
			break
	return round(jaro + prefix * 0.1 * (1 - jaro), 4)


def patient_match_score(
	a: dict[str, Any],
	b: dict[str, Any],
) -> tuple[float, list[str]]:
	"""Probabilistic patient matching using weighted field comparison.

	Input dicts: family, given_0, birth_date (str YYYY-MM-DD), gender,
	national_id, phone, biometric_hash.
	Returns (score 0-1, matching_fields).
	"""
	score = 0.0
	matched: list[str] = []

	if a.get("biometric_hash") and a.get("biometric_hash") == b.get("biometric_hash"):
		score += 0.60
		matched.append("biometric_hash")

	if a.get("national_id") and a.get("national_id") == b.get("national_id"):
		score += 0.35
		matched.append("national_id")

	if a.get("birth_date") and a.get("birth_date") == b.get("birth_date"):
		score += 0.15
		matched.append("birth_date")

	a_fam = (a.get("family") or "").upper()
	b_fam = (b.get("family") or "").upper()
	if a_fam and a_fam == b_fam:
		score += 0.15
		matched.append("family_name_exact")
	elif _soundex(a_fam) == _soundex(b_fam) and a_fam and b_fam:
		jw = _jaro_winkler(a_fam, b_fam)
		if jw >= 0.80:
			score += 0.08
			matched.append("family_name_jaro_winkler")
		else:
			score += 0.04
			matched.append("family_name_soundex")

	a_giv = (a.get("given_0") or "").upper()
	b_giv = (b.get("given_0") or "").upper()
	if a_giv and a_giv == b_giv:
		score += 0.10
		matched.append("given_name_exact")
	elif a_giv and b_giv and _jaro_winkler(a_giv, b_giv) >= 0.85:
		score += 0.05
		matched.append("given_name_fuzzy")

	if a.get("gender") and a.get("gender") == b.get("gender"):
		score += 0.05
		matched.append("gender")

	if a.get("phone") and a.get("phone") == b.get("phone"):
		score += 0.10
		matched.append("phone")

	return round(min(score, 1.0), 3), matched


# ── lab result interpretation ─────────────────────────────────────────────────

def flag_lab_result(
	value: float,
	reference_low: float | None,
	reference_high: float | None,
	critical_low: float | None = None,
	critical_high: float | None = None,
) -> str:
	"""Return LabResultFlag string based on value vs reference/critical ranges."""
	if critical_low is not None and value <= critical_low:
		return "critical_low"
	if critical_high is not None and value >= critical_high:
		return "critical_high"
	if reference_low is not None and value < reference_low:
		return "low"
	if reference_high is not None and value > reference_high:
		return "high"
	return "normal"


def hba1c_to_estimated_avg_glucose_mmol_L(hba1c_pct: float) -> float:
	"""ADA formula: eAG (mmol/L) = (28.7 × HbA1c% − 46.7) / 18.01559."""
	eag_mgdl = 28.7 * hba1c_pct - 46.7
	return round(eag_mgdl / 18.01559, 1)


def gfr_from_cystatin_c(cystatin_c_mg_L: float, age_years: int, is_female: bool) -> float:
	"""CKD-EPI cystatin C equation (2012) — eGFR mL/min/1.73m²."""
	if cystatin_c_mg_L <= 0:
		raise ValueError("cystatin_c must be positive")
	k = 0.8 if cystatin_c_mg_L <= 0.8 else cystatin_c_mg_L
	egfr = 133 * min(cystatin_c_mg_L / 0.8, 1) ** -0.499 * max(cystatin_c_mg_L / 0.8, 1) ** -1.328
	egfr *= 0.996 ** age_years
	if is_female:
		egfr *= 0.932
	return round(egfr, 1)


# ── pharmacy / dosing utilities ────────────────────────────────────────────────

def days_supply(quantity: float, dose_quantity: float, frequency_per_day: float) -> float:
	"""Calculate days supply = quantity / (dose_quantity × frequency_per_day)."""
	if dose_quantity <= 0 or frequency_per_day <= 0:
		raise ValueError("dose_quantity and frequency_per_day must be positive")
	return round(quantity / (dose_quantity * frequency_per_day), 1)


def iv_drip_rate_ml_per_hour(
	total_volume_ml: float,
	infusion_duration_hours: float,
) -> float:
	if infusion_duration_hours <= 0:
		raise ValueError("infusion_duration_hours must be positive")
	return round(total_volume_ml / infusion_duration_hours, 1)


def mg_per_hour_to_mcg_per_kg_per_min(
	mg_per_hour: float,
	weight_kg: float,
) -> float:
	"""Convert IV infusion rate mg/h → mcg/kg/min (vasopressor dosing)."""
	if weight_kg <= 0:
		raise ValueError("weight_kg must be positive")
	return round((mg_per_hour * 1000) / (weight_kg * 60), 2)


# ── NEWS2 sub-score helpers ────────────────────────────────────────────────────

_NEWS2_THRESHOLDS: dict[str, list[tuple[float, int]]] = {
	"respiratory_rate": [(8, 3), (11, 1), (20, 0), (24, 2), (float("inf"), 3)],
	"spo2":             [(91, 3), (93, 2), (95, 1), (float("inf"), 0)],
	"systolic_bp":      [(90, 3), (100, 2), (110, 1), (219, 0), (float("inf"), 3)],
	"heart_rate":       [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (float("inf"), 3)],
	"temperature":      [(35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1), (float("inf"), 2)],
}


def news2_subscore(parameter: str, value: float) -> int:
	"""Return NEWS2 subscore for one vital parameter."""
	thresholds = _NEWS2_THRESHOLDS.get(parameter)
	if thresholds is None:
		raise ValueError(f"Unknown NEWS2 parameter: {parameter}")
	for upper, score in thresholds:
		if value <= upper:
			return score
	return 0
