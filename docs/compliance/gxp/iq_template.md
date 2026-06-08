# Installation Qualification (IQ) Protocol
## APG Platform — GxP System Validation
### FDA 21 CFR Part 11 / EU Annex 11

**Document Number**: APG-VAL-IQ-001  
**Version**: 1.0  
**Status**: DRAFT — requires review by Validation Lead before execution  
**Classification**: Controlled Document

---

## 1. Purpose and Scope

This Installation Qualification (IQ) protocol establishes documented evidence that the
APG Platform has been installed correctly and that the installation meets pre-defined
specifications required for its intended use in regulated pharmaceutical operations.

**System**: APG Platform v[VERSION]  
**Components validated**: All capabilities used in GxP-regulated operations, specifically:
- `pharma_qlt` (Quality Management System)
- `pharma_mfg` (Manufacturing execution data)
- `pharma_ctr` (Clinical Trials Management)
- `pharma_phl` (Pharmacovigilance)
- `pharma_com` (Regulatory Compliance)
- `capabilities/common/esig` (Electronic Signatures — 21 CFR Part 11)
- `capabilities/common/audl` (Immutable Audit Log)

**Regulatory basis**: FDA 21 CFR Part 11, EU GMP Annex 11, GAMP 5

---

## 2. Prerequisites

Before executing this IQ, confirm:

| Prerequisite | Verified By | Date | Result |
|-------------|------------|------|--------|
| Change control CR-[NUMBER] approved | | | |
| System Design Specification (SDS) reviewed | | | |
| Software version matches SDS | | | |
| Infrastructure specification met | | | |
| Installation team trained on GxP documentation | | | |
| APG Platform installation completed | | | |

---

## 3. Installation Verification

### 3.1 Software Version Verification

| Item | Expected | Actual | Pass/Fail |
|------|----------|--------|-----------|
| APG Platform version | v[VERSION] | | |
| Python version | ≥ 3.12.0 | | |
| PostgreSQL version | ≥ 16.0 | | |
| Temporal version | ≥ 1.26 | | |
| NATS JetStream version | ≥ 2.10 | | |
| OPA version | ≥ 0.65 | | |

**Verification method**: `git describe --tags` and service health endpoints

### 3.2 Directory Structure Verification

Confirm that the required directory structure is present:

```
capabilities/
├── common/
│   ├── audl/          [Audit Logging Service]
│   ├── esig/          [Electronic Signatures]
│   ├── phi/           [PHI Classifier]
│   └── vault/         [Tokenization Service]
├── pharma/
│   ├── qlt/           [Quality Management]
│   ├── mfg/           [Manufacturing]
│   ├── ctr/           [Clinical Trials]
│   ├── phl/           [Pharmacovigilance]
│   └── com/           [Compliance]
└── MANIFEST.json      [Capability Registry]
```

| Directory | Present | Correct Permissions | Pass/Fail |
|-----------|---------|--------------------|-----------| 
| capabilities/common/audl/ | | | |
| capabilities/common/esig/ | | | |
| capabilities/pharma/ | | | |
| policies/apg/capabilities/pharma.rego | | | |

### 3.3 Database Schema Verification

Execute the following SQL and verify the output:

```sql
-- Verify audit events table (SOC 2 / GxP)
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_name IN ('apg_audit_events', 'apg_electronic_signatures')
ORDER BY table_name, ordinal_position;
```

| Table | Expected Columns | Verified | Pass/Fail |
|-------|-----------------|----------|-----------|
| apg_audit_events | id, checksum, chain_hash, prev_hash | | |
| apg_electronic_signatures | id, meaning, timestamp, signature_hash | | |

```sql
-- Verify append-only rules (21 CFR Part 11 immutability)
SELECT rulename, qual FROM pg_rules
WHERE tablename IN ('apg_audit_events', 'apg_electronic_signatures');
```

Expected: Rules `apg_audit_events_no_update`, `apg_audit_events_no_delete`,
`apg_esig_no_update`, `apg_esig_no_delete` present.

### 3.4 Electronic Signature Component Verification

Per 21 CFR Part 11.10(g), confirm the three required signature components are captured:

```python
# Execute in APG Python console
from capabilities.common.esig import ESignatureService
svc = ESignatureService(tenant_id="validation")
import asyncio
record = asyncio.run(svc.sign(
    document_id="IQ-TEST-001",
    signer_id="validation.lead@pharma.com",
    meaning="IQ execution verification — this signature confirms installation check",
))
print("Meaning component:", bool(record.meaning))      # Expected: True
print("Identity component:", bool(record.signer_id))   # Expected: True
print("Timestamp component:", bool(record.timestamp))  # Expected: True
print("Hash verification:", record.verify())            # Expected: True
```

| Component | 21 CFR Reference | Expected | Actual | Pass/Fail |
|-----------|-----------------|----------|--------|-----------|
| Meaning | §11.50(a)(3) | Non-empty string | | |
| Signer identity | §11.50(a)(1) | Authenticated user ID | | |
| Timestamp | §11.50(a)(2) | ISO-8601 UTC datetime | | |
| Hash verification | §11.10(e) | True | | |

---

## 4. IQ Summary

| Section | Deviations | Disposition | Initials |
|---------|-----------|------------|---------|
| 3.1 Software versions | | | |
| 3.2 Directory structure | | | |
| 3.3 Database schema | | | |
| 3.4 Electronic signatures | | | |

**Overall IQ result**: [ ] PASS with no deviations  [ ] PASS with deviations  [ ] FAIL

---

## 5. Approvals

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Validation Lead | | | |
| Quality Assurance | | | |
| IT System Owner | | | |

*Proceed to OQ (Operational Qualification) upon IQ approval.*

---

*Document controlled under APG Change Control. Unauthorised amendment is prohibited.*
