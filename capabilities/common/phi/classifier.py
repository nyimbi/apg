"""HIPAA PHI field-level classifier.

Detects Protected Health Information (PHI) in data records by:
  1. Field name matching against HIPAA's 18 PHI identifiers
  2. Value pattern matching (SSN, DOB format, phone, email, etc.)
  3. Context-aware field scoring (medical record context raises sensitivity)

Based on HIPAA Safe Harbor Method (45 CFR §164.514(b)) — the 18 identifiers
that must be removed to de-identify health information.

Reference: https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# HIPAA's 18 identifiers mapped to common field name patterns
_PHI_FIELD_PATTERNS: dict[str, list[str]] = {
	"name":               ["name", "first_name", "last_name", "full_name", "patient_name", "given_name", "surname"],
	"geographic":         ["address", "street", "city", "zip", "postal_code", "geo", "location", "county", "region"],
	"date":               ["dob", "date_of_birth", "birth_date", "birthdate", "admission_date", "discharge_date", "death_date"],
	"phone":              ["phone", "telephone", "mobile", "cell", "fax", "contact_number"],
	"fax":                ["fax", "fax_number"],
	"email":              ["email", "email_address", "e_mail"],
	"ssn":                ["ssn", "social_security", "national_id", "national_number", "id_number"],
	"mrn":                ["mrn", "medical_record", "patient_id", "encounter_id", "visit_id"],
	"health_plan":        ["health_plan", "insurance_id", "member_id", "plan_id", "policy_number"],
	"account":            ["account_number", "account_id", "bank_account"],
	"certificate":        ["certificate", "license", "registration_number"],
	"vehicle":            ["vehicle_id", "vin", "license_plate", "plate_number"],
	"device":             ["device_id", "device_serial", "serial_number", "imei"],
	"url":                ["profile_url", "personal_url", "homepage"],
	"ip_address":         ["ip_address", "ip_addr", "client_ip"],
	"biometric":          ["fingerprint", "biometric", "retina", "iris", "voiceprint", "dna"],
	"photo":              ["photo", "image", "picture", "face_image", "portrait"],
	"clinical":           ["diagnosis", "icd_code", "cpt_code", "condition", "treatment", "medication",
	                       "prescription", "lab_result", "clinical_note", "symptom", "allergy",
	                       "procedure", "surgery", "test_result"],
}

# Flat set of all PHI field name fragments (lower-cased)
_PHI_FRAGMENTS: set[str] = {
	frag for patterns in _PHI_FIELD_PATTERNS.values() for frag in patterns
}

# Value-level patterns for PHI detection
_PHI_VALUE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
	("ssn",   re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")),
	("dob",   re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
	("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")),
	("phone", re.compile(r"\b(\+?254|0)[17]\d{8}\b|\b\+?[1-9]\d{6,14}\b")),
	("mrn",   re.compile(r"\b(MRN|P)[- ]?\d{4,10}\b", re.IGNORECASE)),
]

# Minimum-necessary field sets by purpose (HIPAA §164.502(b))
_MINIMUM_NECESSARY: dict[str, set[str]] = {
	"treatment":  {"name", "dob", "mrn", "clinical", "date", "phone"},
	"payment":    {"name", "health_plan", "account", "date"},
	"operations": {"mrn", "date", "clinical"},
	"billing":    {"name", "health_plan", "account", "date", "geographic"},
	"research":   {"date", "clinical"},  # de-identified cohort only
}


@dataclass
class PHIClassificationResult:
	"""Result of PHI classification on a data record."""
	contains_phi: bool
	phi_fields: list[str]         # field names detected as PHI
	phi_categories: list[str]     # HIPAA identifier categories present
	risk_level: str               # "none" | "low" | "medium" | "high" | "critical"
	minimum_necessary: dict[str, list[str]]  # purpose -> allowed phi_fields


def phi_fields(data: dict[str, Any]) -> list[str]:
	"""Return field names in data that are likely PHI."""
	return PHIClassifier().classify(data).phi_fields


def is_phi_field(field_name: str) -> bool:
	"""Return True if a field name is likely to contain PHI."""
	lower = field_name.lower()
	return any(frag in lower for frag in _PHI_FRAGMENTS)


class PHIClassifier:
	"""HIPAA PHI field-level classifier.

	Detects PHI by matching field names against HIPAA's 18 identifiers and
	value patterns (SSN, email, phone, etc.). Returns a detailed classification
	result with risk level and minimum-necessary field sets by purpose.
	"""

	def classify(self, data: dict[str, Any]) -> PHIClassificationResult:
		"""Classify a data dict for PHI content.

		Args:
			data: A flat or nested dict of field names to values.

		Returns:
			PHIClassificationResult with per-field PHI tagging.
		"""
		detected_fields: list[str] = []
		detected_categories: set[str] = set()

		for field_name, value in self._flatten(data).items():
			lower_name = field_name.lower()

			# Field name matching
			matched_by_name = False
			for category, patterns in _PHI_FIELD_PATTERNS.items():
				if any(p in lower_name for p in patterns):
					detected_fields.append(field_name)
					detected_categories.add(category)
					matched_by_name = True
					break

			# Value pattern matching (for fields not caught by name)
			if not matched_by_name and value is not None:
				str_val = str(value)
				for category, pattern in _PHI_VALUE_PATTERNS:
					if pattern.search(str_val):
						detected_fields.append(field_name)
						detected_categories.add(category)
						break

		risk = self._risk_level(detected_categories)
		min_necessary = self._minimum_necessary(detected_fields, detected_categories)

		return PHIClassificationResult(
			contains_phi=bool(detected_fields),
			phi_fields=sorted(set(detected_fields)),
			phi_categories=sorted(detected_categories),
			risk_level=risk,
			minimum_necessary=min_necessary,
		)

	def redact(
		self,
		data: dict[str, Any],
		purpose: str = "operations",
		replacement: str = "[REDACTED]",
	) -> dict[str, Any]:
		"""Return a copy of data with PHI fields outside the minimum-necessary set redacted.

		Args:
			data: Original data dict
			purpose: HIPAA purpose (treatment/payment/operations/billing/research)
			replacement: Value to use for redacted fields
		"""
		result = self.classify(data)
		allowed = _MINIMUM_NECESSARY.get(purpose, set())
		allowed_categories = allowed  # purpose maps to categories

		redacted = dict(data)
		for f in result.phi_fields:
			# Check if this field's categories are allowed for the purpose
			field_cats = {
				cat for cat, patterns in _PHI_FIELD_PATTERNS.items()
				if any(p in f.lower() for p in patterns)
			}
			if not field_cats.intersection(allowed_categories):
				redacted[f] = replacement

		return redacted

	# ── private ──────────────────────────────────────────────────────────

	@staticmethod
	def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
		"""Flatten nested dicts with dot-notation keys."""
		flat: dict[str, Any] = {}
		for k, v in data.items():
			key = f"{prefix}.{k}" if prefix else k
			if isinstance(v, dict):
				flat.update(PHIClassifier._flatten(v, key))
			else:
				flat[key] = v
		return flat

	@staticmethod
	def _risk_level(categories: set[str]) -> str:
		if not categories:
			return "none"
		critical = {"ssn", "biometric", "mrn"}
		high = {"clinical", "health_plan", "date", "name", "dob"}
		medium = {"email", "phone", "address", "geographic"}
		if categories & critical:
			return "critical"
		if categories & high:
			return "high"
		if categories & medium:
			return "medium"
		return "low"

	@staticmethod
	def _minimum_necessary(
		phi_fields_list: list[str],
		categories: set[str],
	) -> dict[str, list[str]]:
		"""Map each HIPAA access purpose to the PHI fields allowed for it."""
		result: dict[str, list[str]] = {}
		for purpose, allowed_cats in _MINIMUM_NECESSARY.items():
			allowed: list[str] = []
			for f in phi_fields_list:
				field_cats = {
					cat for cat, patterns in _PHI_FIELD_PATTERNS.items()
					if any(p in f.lower() for p in patterns)
				}
				if field_cats.intersection(allowed_cats):
					allowed.append(f)
			result[purpose] = sorted(allowed)
		return result
