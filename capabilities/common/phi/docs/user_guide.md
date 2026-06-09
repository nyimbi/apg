# PHI Classifier — HIPAA Compliance User Guide

## Overview

The `phi` capability detects and redacts the 18 HIPAA Safe Harbor identifiers from data records and documents. It integrates transparently into healthcare and other data pipelines to enforce minimum-necessary access and maintain HIPAA compliance.

## HIPAA 18 Identifiers Detected

Names, geographic subdivisions, dates, phone numbers, fax numbers, email addresses, SSNs, medical record numbers, health plan beneficiary numbers, account numbers, certificate/license numbers, vehicle identifiers, device identifiers, URLs, IP addresses, biometric identifiers, full-face photos, and any other unique identifying number.

## Classify a Field

```python
from capabilities.common.phi.service import PHIService

svc = PHIService(tenant_id="hospital_a")
result = await svc.classify("patient_ssn", "123-45-6789")
# {"field_name": "patient_ssn", "is_phi": True, "identifier_type": "ssn", "confidence": 0.9}
```

## Redact a Record

```python
record = {
    "patient_name": "Jane Doe",
    "diagnosis": "J18.9",
    "email": "jane@example.com",
    "temperature": 38.2,
}
result = await svc.redact(record)
# {
#   "redacted_record": {"patient_name": "[REDACTED]", "diagnosis": "J18.9", "email": "[REDACTED]", "temperature": 38.2},
#   "phi_fields_found": ["patient_name", "email"],
#   "phi_count": 2,
#   "total_fields": 4,
# }
```

## Scan a Document

```python
result = await svc.scan_document("Patient Jane Doe (SSN: 123-45-6789) admitted on 2024-01-10.")
# {"findings": [{"type": "SSN", "position": 24, "value": "REDACTED"}], "phi_count": 1}
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/phi/classify` | Classify a single field |
| POST | `/api/phi/classify/batch` | Classify multiple fields |
| POST | `/api/phi/redact` | Redact PHI from a record |
| POST | `/api/phi/redact/batch` | Redact PHI from multiple records |
| POST | `/api/phi/scan/record` | Scan a record dict |
| POST | `/api/phi/scan/document` | Scan free-text |
| GET | `/api/phi/identifiers` | List monitored identifiers |
| POST | `/api/phi/identifiers/test` | Test a regex pattern |
| POST | `/api/phi/access/log` | Log PHI access event |
| GET | `/api/phi/compliance` | Compliance status |
| POST | `/api/phi/validate/deidentification` | Validate de-identification |
| GET | `/api/phi/audit` | PHI access audit log |
| GET | `/api/phi/report` | PHI compliance report |
| GET | `/api/phi/health` | Health check |

## Minimum Necessary Enforcement

PHI access is governed by purpose (treatment/payment/operations/research). The classifier returns a `minimum_necessary` map specifying which PHI fields are allowed per purpose — OPA policies use this to enforce access at the API boundary.

## Integration with Healthcare Capabilities

`healthcare_emr`, `healthcare_cli`, and `healthcare_img` automatically route records through the PHI classifier when `phi.scan_record()` is called in their service layer. Set `PHI_AUDIT_ENABLED=true` to log all PHI access to the NATS audit stream.
