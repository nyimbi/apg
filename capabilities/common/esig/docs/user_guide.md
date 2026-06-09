# Electronic Signature — FDA 21 CFR Part 11 User Guide

## Overview

The `esig` capability implements qualified electronic signatures meeting FDA 21 CFR Part 11. Every signature cryptographically binds the three required components: signer intent (meaning), signer identity, and timestamp. Designed for pharmaceutical batch records, laboratory notebooks, change control, and any regulated document requiring an audit-defensible signature.

## 21 CFR Part 11 Three-Component Requirement

Per 21 CFR Part 11.50(a), every electronic signature must capture:

1. **Meaning** — the signer's stated intent (e.g., "I approve this batch record for release")
2. **Identity** — the authenticated signer (user ID, not ambiguous)
3. **Date/Time** — UTC timestamp of signing

The `esig` service SHA-256 hashes `document_id:meaning:signer_id:timestamp` to produce a tamper-evident signature binding all three.

## Signing a Document

```python
from capabilities.common.esig.service import ESignatureService

svc = ESignatureService(tenant_id="pharma_corp")

record = await svc.sign(
    document_id="batch_record_BR-2024-001",
    signer_id="qa_manager@pharma.com",
    signer_display_name="Dr. Sarah Chen",
    meaning="I certify this batch record is accurate and complete per SOP-QA-001",
)

print(record.signature_id)     # UUID7
print(record.signature_hash)   # SHA-256 binding
print(record.timestamp)        # ISO-8601 UTC
```

## Verifying a Signature

```python
result = await svc.verify(record.signature_id)
# {"valid": True, "verified_at": "2024-01-10T14:32:11Z"}
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/esig/sign` | Sign a document |
| POST | `/api/esig/sign/batch` | Sign multiple documents |
| POST | `/api/esig/verify` | Verify a signature |
| GET | `/api/esig/signatures/{document_id}` | List signatures on a document |
| DELETE | `/api/esig/signatures/{signature_id}` | Revoke a signature |
| GET | `/api/esig/compliance` | 21 CFR Part 11 compliance report |
| GET | `/api/esig/audit` | Full audit trail |
| GET | `/api/esig/health` | Health check |

## GxP Use Cases

- **Batch Record Release** (`pharma_qms`): Signs batch records with QA approval
- **Lab Notebook** (`pharma_lims`): Signs analytical results
- **Change Control** (`ckm_wfa`): Signs change records per SOP
- **Clinical Protocol** (`pharma_ctm`): Signs protocol amendments per ICH E6(R2)

## Signature Storage

In development, signatures are stored in memory. In production, inject a SQLAlchemy session:

```python
from sqlalchemy.ext.asyncio import AsyncSession
svc = ESignatureService(tenant_id="pharma_corp", db=async_session)
```

The service persists to `apg_electronic_signatures` (see `0001_electronic_signatures.sql`) with SHA-256 hash chain for tamper evidence.
