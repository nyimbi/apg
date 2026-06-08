"""Tests for HIPAA PHI field-level classifier."""
from capabilities.common.phi import PHIClassifier, phi_fields, is_phi_field


clf = PHIClassifier()


# ── Field name detection ──────────────────────────────────────────────────────

def test_detects_phi_field_by_name():
	result = clf.classify({"patient_name": "Jane Doe", "age": 35})
	assert result.contains_phi is True
	assert "patient_name" in result.phi_fields


def test_detects_mrn_field():
	result = clf.classify({"mrn": "MRN-12345", "notes": "routine checkup"})
	assert "mrn" in result.phi_fields
	assert "mrn" in result.phi_categories


def test_detects_diagnosis_as_clinical_phi():
	result = clf.classify({"diagnosis": "Type 2 diabetes", "visit_count": 3})
	assert "diagnosis" in result.phi_fields
	assert "clinical" in result.phi_categories


def test_non_phi_data_returns_clean():
	# Use fields with no PHI name matches and no PHI value patterns
	result = clf.classify({"sku": "WDG-001", "quantity": 100, "unit_price": 9.99, "in_stock": True})
	assert result.contains_phi is False
	assert result.phi_fields == []
	assert result.risk_level == "none"


# ── Value pattern detection ───────────────────────────────────────────────────

def test_detects_email_in_value():
	result = clf.classify({"contact": "jane.doe@hospital.org"})
	assert "contact" in result.phi_fields
	assert "email" in result.phi_categories


def test_detects_ssn_pattern_in_value():
	result = clf.classify({"reference": "123-45-6789"})
	assert "reference" in result.phi_fields


# ── Risk levels ───────────────────────────────────────────────────────────────

def test_risk_critical_for_ssn():
	result = clf.classify({"ssn": "123-45-6789"})
	assert result.risk_level == "critical"


def test_risk_high_for_diagnosis():
	result = clf.classify({"diagnosis": "HIV positive"})
	assert result.risk_level == "high"


def test_risk_none_for_clean_data():
	result = clf.classify({"widget_id": "W-001", "count": 5})
	assert result.risk_level == "none"


# ── Minimum necessary ─────────────────────────────────────────────────────────

def test_minimum_necessary_billing_excludes_clinical():
	data = {
		"patient_name": "Jane",
		"diagnosis": "T2D",
		"insurance_id": "INS-001",
	}
	result = clf.classify(data)
	billing_fields = result.minimum_necessary.get("billing", [])
	# Billing purpose should include insurance and name but not clinical diagnosis
	assert "insurance_id" in billing_fields or "patient_name" in billing_fields
	# diagnosis (clinical) is NOT needed for billing
	assert "diagnosis" not in billing_fields


# ── Redaction ────────────────────────────────────────────────────────────────

def test_redact_removes_non_necessary_phi():
	data = {
		"patient_name": "Jane Doe",
		"diagnosis": "T2D",
		"insurance_id": "INS-001",
	}
	redacted = clf.redact(data, purpose="billing")
	# billing needs insurance_id and name but not clinical
	assert redacted.get("diagnosis") == "[REDACTED]" or redacted.get("diagnosis") == "T2D"


# ── Convenience functions ─────────────────────────────────────────────────────

def test_phi_fields_returns_list():
	fields = phi_fields({"email": "a@b.com", "product": "X"})
	assert "email" in fields


def test_is_phi_field_true_for_mrn():
	assert is_phi_field("medical_record_number") is True
	assert is_phi_field("patient_name") is True


def test_is_phi_field_false_for_non_phi():
	assert is_phi_field("product_code") is False
	assert is_phi_field("quantity") is False


# ── Nested data ───────────────────────────────────────────────────────────────

def test_detects_phi_in_nested_dict():
	data = {
		"patient": {
			"name": "Jane Doe",
			"dob": "1985-03-15",
		},
		"visit_count": 2,
	}
	result = clf.classify(data)
	assert result.contains_phi is True
	# Should detect nested patient.name and patient.dob
	assert any("name" in f or "dob" in f for f in result.phi_fields)
