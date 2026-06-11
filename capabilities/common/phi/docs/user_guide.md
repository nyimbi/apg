# PHI Classifier — HIPAA Compliance User Guide

## Overview

The `phi` capability detects and redacts the 18 HIPAA Safe Harbor identifiers from data
records, free-text documents, database schemas, and FHIR R4 resources. It integrates into
healthcare and data pipelines to enforce minimum-necessary access, provide risk scoring,
pseudonymisation, differential privacy, and streaming redaction.

## HIPAA 18 Identifiers Detected

Names, geographic subdivisions, dates, phone numbers, fax numbers, email addresses, SSNs,
medical record numbers, health plan beneficiary numbers, account numbers,
certificate/license numbers, vehicle identifiers, device identifiers, URLs, IP addresses,
biometric identifiers, full-face photos, and any other unique identifying number
(45 CFR §164.514(b)).

---

## Core Usage

### Classify a Single Field

```python
from capabilities.common.phi.service import PHIService

svc = PHIService(tenant_id="hospital_a")
result = await svc.classify("patient_ssn", "123-45-6789")
# {
#   "field_name": "patient_ssn",
#   "is_phi": True,
#   "identifier_type": "ssn",
#   "confidence": 0.9,
#   "regulation": "HIPAA",
# }
```

### Redact a Record

```python
record = {
    "patient_name": "Jane Doe",
    "diagnosis": "J18.9",
    "email": "jane@example.com",
    "temperature": 38.2,
}
result = await svc.redact(record)
# {
#   "redacted_record": {
#     "patient_name": "[REDACTED]",
#     "diagnosis": "J18.9",
#     "email": "[REDACTED]",
#     "temperature": 38.2,
#   },
#   "phi_fields_found": ["email", "patient_name"],
#   "phi_count": 2,
#   "total_fields": 4,
# }
```

### Scan a Record Without Redacting

```python
scan = await svc.scan_record(record)
# {
#   "phi_fields": [{"field_name": "patient_name", ...}, {"field_name": "email", ...}],
#   "phi_count": 2,
#   "total_fields": 4,
#   "phi_density": 0.5,
# }
```

### Batch Operations

```python
fields = [{"field_name": "ssn", "value": "123-45-6789"}, {"field_name": "age", "value": "42"}]
results = await svc.classify_batch(fields)

records = [record_a, record_b, record_c]
redacted_list = await svc.redact_batch(records)
```

---

## Document Scanning and Redaction

### Basic Scan (3 patterns)

```python
result = await svc.scan_document("Patient Jane Doe (SSN: 123-45-6789) admitted on 2024-01-10.")
# {"findings": [{"type": "SSN", "position": 24, "value": "REDACTED"}], "phi_count": 1}
```

### Full Scan with Span-Level Output (10 patterns)

Covers SSN, PHONE, EMAIL, DOB, MRN, IP, URL, ZIP, CREDIT_CARD, NPI with character
offsets and context snippets for highlighting UIs and audit logs.

```python
result = await svc.scan_document_full("Contact: jane@example.com | IP: 192.168.1.10")
# {
#   "findings": [
#     {"type": "EMAIL", "start": 9, "end": 25, "context_snippet": "Contact: [REDACTED] | IP: 19", ...},
#     {"type": "IP", ...},
#   ],
#   "phi_count": 2,
#   "categories_found": ["EMAIL", "IP"],
#   "text_length": 44,
#   "phi_density": 0.4,
# }
```

### Redact a Document (Full)

```python
out = await svc.redact_document(text, replacement="[PHI]")
# {
#   "redacted_text": "Contact: [PHI] | IP: [PHI]",
#   "original_length": 44,
#   "redacted_length": 26,
#   "replacements": 2,
#   "change_log": [{"type": "EMAIL", "start": 9, ...}, ...],
# }
```

---

## Risk Scoring

### Composite PHI Risk Score

Produces a normalised `[0.0, 1.0]` risk score and a `risk_band` label
(none / low / medium / high / critical) based on category criticality weights.

```python
result = await svc.score_phi_risk({
    "ssn": "123-45-6789",
    "mrn": "MRN-0012345",
    "temperature": 38.2,
})
# {"risk_score": 0.6667, "risk_band": "high", "phi_categories": ["mrn", "ssn"], ...}
```

### Re-identification Risk (k-Anonymity)

Groups a cohort on quasi-identifiers (age, gender, zip, race) and returns the
minimum equivalence class size (k-value). k < 5 is high risk.

```python
cohort = [
    {"age": 35, "gender": "F", "zip": "10001", "diagnosis": "E11.9"},
    {"age": 35, "gender": "F", "zip": "10001", "diagnosis": "I10"},
]
result = await svc.score_reidentification_risk(cohort)
# {"k_value": 2, "risk_summary": "HIGH risk: smallest equivalence class has 2 records ...", ...}
```

---

## Pseudonymisation

Replace PHI values with deterministic HMAC-derived pseudonyms. Store the
`pseudonym_map` in a secrets manager for later re-identification by authorised users.

```python
import os

out = await svc.pseudonymise(record, namespace="hospital_a", secret=os.environ["PHI_SECRET"])
# {
#   "pseudonymised_record": {"patient_name": "pseudo_3f8a...", "ssn": "pseudo_91bc...", ...},
#   "pseudonym_map": {"Jane Doe": "pseudo_3f8a...", "123-45-6789": "pseudo_91bc..."},
#   "phi_fields_pseudonymised": ["patient_name", "ssn"],
# }

# Re-identification (authorised path only)
restored = await svc.depseudonymise(out["pseudonymised_record"], out["pseudonym_map"])
```

---

## Schema-Level Classification

Classify a database schema without row data — for data catalogs and migration guards.

```python
schema = {
    "patient_name": "varchar(255)",
    "date_of_birth": "date",
    "temperature":   "float",
    "ssn":           "char(11)",
}
result = await svc.classify_schema(schema)
# {
#   "phi_column_count": 3,
#   "phi_fraction": 0.75,
#   "phi_columns": ["patient_name", "date_of_birth", "ssn"],
#   "columns": [{"column": "patient_name", "is_phi": True, "hipaa_label": "Name", ...}, ...],
# }
```

---

## Streaming Redaction

For high-throughput ETL pipelines that cannot load datasets into memory.

```python
async def source_rows():
    async for row in db.execute("SELECT * FROM patients"):
        yield dict(row)

async for clean_row in svc.redact_stream(source_rows(), batch_size=100):
    await sink.insert(clean_row)
```

---

## FHIR R4 Classification

Classify FHIR R4 resources using resource-type-aware path mapping.

```python
patient = {
    "resourceType": "Patient",
    "name": [{"use": "official", "family": "Doe", "given": ["Jane"]}],
    "birthDate": "1990-05-15",
    "telecom": [{"system": "phone", "value": "555-1234"}],
}
result = await svc.classify_fhir_resource(patient)
# {
#   "resource_type": "Patient",
#   "fhir_path_findings": [
#     {"fhir_path": "Patient.name", "hipaa_category": "name", ...},
#     {"fhir_path": "Patient.birthDate", "hipaa_category": "date", ...},
#     {"fhir_path": "Patient.telecom", "hipaa_category": "phone", ...},
#   ],
#   "risk_level": "high",
# }
```

Supported resource types: Patient, Encounter, Observation, DiagnosticReport.

---

## OPA Policy Export

Generate Open Policy Agent Rego policies from the HIPAA minimum-necessary map
to enforce field-level access at the API gateway boundary.

```python
opa = await svc.export_opa_policy(purpose="treatment")
# {
#   "rego_policy": "package phi.minimum_necessary\n...",
#   "policy_filename": "phi_minimum_necessary.rego",
#   "purposes_exported": ["treatment"],
# }
# Write to OPA bundle:
with open("phi_minimum_necessary.rego", "w") as f:
    f.write(opa["rego_policy"])
```

---

## Synthetic PHI Test Data

Generate statistically realistic HIPAA-profile records for unit tests, load tests,
and demos — no real patient data used.

```python
out = await svc.generate_synthetic_phi_record(locale="en_US", seed=42)
record = out["record"]        # flat PHI dict with all 18 identifiers
fhir   = out["fhir_patient"]  # FHIR R4 Patient bundle

# Reproducible — same seed produces same record:
out2 = await svc.generate_synthetic_phi_record(seed=42)
assert out["record"]["ssn"] == out2["record"]["ssn"]
```

---

## Access Velocity Anomaly Detection

Track PHI access rate per accessor. Raises `is_anomalous` if threshold exceeded.

```python
result = await svc.check_phi_access_velocity(
    accessor_id="user_123",
    window_seconds=300,   # 5-minute rolling window
    threshold=100,
)
# {
#   "accessor_id": "user_123",
#   "access_count": 12,
#   "is_anomalous": False,
#   "recommendation": "Normal: 12 accesses within 300s window.",
# }
```

Production deployments should back this with Redis sorted sets and emit NATS
`phi.access.anomaly` events when `is_anomalous` is True.

---

## Differential Privacy

Apply Laplace-mechanism noise to aggregate statistics derived from PHI data.
Meets HIPAA Expert Determination requirements for releasing statistics.

```python
counts = [150, 300, 75, 220]
result = await svc.apply_laplace_noise(counts, epsilon=0.5, sensitivity=1.0)
# {
#   "noisy_values": [151.3, 298.7, 76.1, 219.4],
#   "epsilon": 0.5,
#   "scale": 2.0,
#   "privacy_interpretation": "Moderate privacy (0.1 <= epsilon < 1.0) — good trade-off.",
# }
```

Smaller epsilon = stronger privacy guarantee, larger noise. Use epsilon < 1.0 for
publishing aggregate PHI statistics externally.

---

## Per-Category Breakdown

```python
result = await svc.get_phi_category_breakdown(record)
# {
#   "field_breakdown": [
#     {"field_name": "ssn", "hipaa_category": "ssn", "hipaa_label": "Social Security Number", "risk_weight": 1.0},
#     {"field_name": "patient_name", "hipaa_category": "name", "hipaa_label": "Name", "risk_weight": 0.7},
#   ],
#   "category_summary": {"ssn": ["ssn"], "name": ["patient_name"]},
#   "risk_level": "critical",
#   "phi_count": 2,
# }
```

---

## Compliance Features

### Minimum Necessary Enforcement

PHI access is governed by purpose (treatment / payment / operations / billing / research).
The classifier returns a `minimum_necessary` map specifying which PHI fields are allowed
per purpose — used by OPA policies enforced at the API gateway.

```python
scan = await svc.validate_minimum_necessary(record, purpose="treatment", role="physician")
deident = await svc.validate_deidentification(record)
cert = await svc.certify_safe_harbor(record)
```

### Business Associate Agreement Check

```python
baa = await svc.check_baa_requirement("export_phi")
# {"operation": "export_phi", "requires_baa": True}
```

### Compliance Status and Report

```python
status = await svc.get_compliance_status()
report = await svc.generate_phi_report()
```

---

## Integration with Healthcare Capabilities

`healthcare_emr`, `healthcare_cli`, and `healthcare_img` automatically route records
through the PHI classifier when `phi.scan_record()` is called in their service layer.
Set `PHI_AUDIT_ENABLED=true` to log all PHI access to the NATS audit stream.

Set `PHI_SECRET` environment variable to the HMAC key used for pseudonymisation.
Rotate it via `pseudonymise` with a new secret and `depseudonymise` with the old one.
