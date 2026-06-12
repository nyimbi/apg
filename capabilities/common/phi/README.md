# APG PHI Classifier (HIPAA) (`phi`)

**Version**: 2.0.0 | **Domain**: common

## Overview

Production-grade HIPAA PHI detection, redaction, and compliance enforcement.
Detects all 18 HIPAA Safe Harbor identifiers by field name and value pattern.
Provides risk scoring, pseudonymisation, streaming redaction, FHIR R4 classification,
k-anonymity analysis, OPA policy export, synthetic test-data generation, velocity
anomaly detection, differential privacy noise injection, and schema-level classification.

## Core Capabilities

| Feature | Method(s) |
|---|---|
| Single-field classify | `classify(field_name, value)` |
| Record redact | `redact(record)` |
| Batch classify / redact | `classify_batch`, `redact_batch` |
| Record / document scan | `scan_record`, `scan_document`, `scan_document_full` |
| Document redaction (full) | `redact_document(text)` |
| Query result scan | `scan_query_result(rows)` |
| Log masking | `mask_phi_in_logs(log_entry)` |
| PHI risk scoring | `score_phi_risk(record)` |
| k-anonymity / re-ID risk | `score_reidentification_risk(cohort)` |
| Pseudonymisation | `pseudonymise / depseudonymise` |
| Schema classification | `classify_schema(schema)` |
| Streaming redaction | `redact_stream(async_iterator)` |
| FHIR R4 classification | `classify_fhir_resource(resource)` |
| OPA Rego policy export | `export_opa_policy(purpose)` |
| Synthetic test data | `generate_synthetic_phi_record(locale, seed)` |
| Access velocity alert | `check_phi_access_velocity(accessor_id)` |
| Differential privacy | `apply_laplace_noise(values, epsilon)` |
| Category breakdown | `get_phi_category_breakdown(record)` |

## Quick Start

```python
from capabilities.common.phi.service import PHIService

svc = PHIService(tenant_id="hospital_a")

# Classify a single field
result = await svc.classify("patient_ssn", "123-45-6789")
# {"field_name": "patient_ssn", "is_phi": True, "identifier_type": "ssn", "confidence": 0.9}

# Redact a record
result = await svc.redact({"patient_name": "Jane Doe", "temperature": 38.2})

# Scan free text with full span data
result = await svc.scan_document_full("Patient Jane Doe SSN: 123-45-6789")

# Risk score
result = await svc.score_phi_risk({"ssn": "123-45-6789", "mrn": "MRN-001"})
# {"risk_score": 1.0, "risk_band": "critical", ...}

# Pseudonymise and restore
out = await svc.pseudonymise(record, namespace="hospital_a", secret=os.environ["PHI_SECRET"])
restored = await svc.depseudonymise(out["pseudonymised_record"], out["pseudonym_map"])

# Schema-level guard
await svc.classify_schema({"patient_name": "varchar", "temperature": "float"})

# FHIR R4 patient resource
await svc.classify_fhir_resource({"resourceType": "Patient", "name": [...], "birthDate": "1990-01-01"})

# Export OPA Rego policy
opa = await svc.export_opa_policy(purpose="treatment")

# Differential privacy on aggregate counts
dp = await svc.apply_laplace_noise([100, 250, 75], epsilon=0.5)

# Streaming ETL redaction
async def source():
    for row in db.cursor:
        yield row

async for clean_row in svc.redact_stream(source(), batch_size=100):
    await sink.write(clean_row)
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/phi/classify` | Classify a single field |
| POST | `/api/phi/classify/batch` | Classify multiple fields |
| POST | `/api/phi/redact` | Redact PHI from a record |
| POST | `/api/phi/redact/batch` | Redact PHI from multiple records |
| POST | `/api/phi/scan/record` | Scan a record dict |
| POST | `/api/phi/scan/document` | Scan free-text (3-pattern) |
| POST | `/api/phi/scan/document/full` | Scan free-text (10-pattern, span-level) |
| POST | `/api/phi/redact/document` | Redact free-text document |
| POST | `/api/phi/risk/score` | Composite PHI risk score |
| POST | `/api/phi/risk/reidentification` | k-anonymity cohort risk |
| POST | `/api/phi/pseudonymise` | Pseudonymise PHI fields |
| POST | `/api/phi/depseudonymise` | Reverse pseudonymisation |
| POST | `/api/phi/schema/classify` | Classify column schema |
| POST | `/api/phi/fhir/classify` | Classify FHIR R4 resource |
| POST | `/api/phi/opa/export` | Export OPA Rego policy |
| POST | `/api/phi/synthetic` | Generate synthetic PHI record |
| POST | `/api/phi/velocity` | Check access velocity |
| POST | `/api/phi/privacy/laplace` | Apply Laplace DP noise |
| POST | `/api/phi/breakdown` | Per-category PHI breakdown |
| GET | `/api/phi/identifiers` | List monitored identifiers |
| POST | `/api/phi/identifiers/test` | Test a regex pattern |
| POST | `/api/phi/access/log` | Log PHI access event |
| GET | `/api/phi/compliance` | Compliance status |
| POST | `/api/phi/validate/deidentification` | Validate de-identification |
| GET | `/api/phi/audit` | PHI access audit log |
| GET | `/api/phi/report` | PHI compliance report |
| GET | `/api/phi/health` | Health check |

## Governance Rules

- tenant_context_required
- operation_type_required
- audit_logged
- access_controlled

## License

© 2025 Datacraft | nyimbi@gmail.com | www.datacraft.co.ke

---

## World-Class Enhancements (v2.0)

- **I1.** PHI Classifier — 15 World-Class Improvements
- **I2.** Probabilistic Confidence Scoring (Multi-Signal Bayesian)
- **I3.** NLP-Backed Free-Text PHI Extraction (NER via Ollama)
- **I4.** Streaming Redaction Pipeline
- **I5.** Structured Audit Ledger (Append-Only Event Store)
- **I6.** Field-Level Encryption (Format-Preserving AES-FF3)
- **I7.** Differential Privacy Noise Injection
- **I8.** De-identification Quality Metrics (Re-identification Risk Score)
- **I9.** FHIR R4 Resource PHI Extractor
- **I10.** Schema-Aware Column Classifier (SQL/Parquet/Arrow)
- **I11.** Redaction Policy DSL
- **I12.** Provenance-Preserving Reversible Pseudonymisation
- **I13.** Cross-Tenant PHI Linkage Detection
- **I14.** HIPAA Minimum-Necessary OPA Policy Export
- **I15.** Real-Time PHI Velocity Alerting

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
