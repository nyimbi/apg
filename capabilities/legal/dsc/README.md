# leg_dsc — Document & eDiscovery

Document repository, version control, privilege logging, litigation hold, and eDiscovery production with rolling Bates numbering, review coding, redaction engine, FRCP deadline calendar, forensic integrity verification, document families, retention/destruction policy, hold acknowledgement workflow, privilege challenge tracker, and matter cost tracking.

## Features

### Core Repository
- Document ingestion with MIME type, version control, and full audit trail
- Soft-delete (archive) with hold enforcement
- Full-text search across title and description fields

### Forensic Integrity (I4)
- SHA-256 content hash stored at ingest
- `verify_integrity()` — tamper-detection with court-admissible audit log entry

### Review Coding + Near-Dup Propagation (I5)
- `code_document()` — records responsive/non_responsive/redact/withhold decisions
- `propagate_coding()` — auto-codes near-duplicate cluster at 0.85 confidence, cutting re-review cost

### Redaction Engine (I6)
- `add_redaction()` — records page/bbox/reason/redactor per redaction
- `list_redactions()` — returns structured redaction manifest for privilege-log export

### FRCP Discovery Deadline Calendar (I7)
- `create_discovery_deadline()` — tracks rule26_initial, meet_and_confer, production_response, etc.
- `list_overdue_deadlines()` — returns deadlines past due with `days_overdue` computed

### Privilege Challenge Tracker (I8)
- `raise_privilege_challenge()` — opens formal challenge against privilege log entry
- `respond_to_challenge()` — records formal response with supporting documents
- `rule_on_challenge()` — records court ruling; auto-removes privilege on overruled decisions

### Rolling Bates Numbering (I9)
- Production sets start Bates numbering from the prior matter high-water mark
- `get_bates_range()` — returns current high-water mark and next start number

### Document Families (I10)
- `attach_document()` — links child documents (attachments) to parent (FRCP Rule 34(b)(2)(E))
- `get_document_family()` — returns all family members in parent-first order

### Retention & Destruction Policy (I11)
- `set_retention_policy()` — attaches policy_id and destroy_after_date to documents
- `list_destruction_eligible()` — returns docs past retention date that are NOT on hold

### Matter Cost Tracking (I12)
- `record_cost()` — records processing/hosting/review/production costs as Decimal (no float)
- `matter_cost_summary()` — returns totals by cost_type and grand total, all as Decimal strings

### Hold Acknowledgement Workflow (I2)
- `request_hold_acknowledgement()` — creates a formal ack request with due date
- `record_acknowledgement()` — records custodian receipt with signature reference
- `list_unacknowledged_holds()` — returns overdue acknowledgements for escalation

### Privilege Log
- Auto-logged on privileged document upload
- Manual `log_privilege()` endpoint

### Litigation Holds
- Auto-applies hold to all matching matter documents
- Scope query for keyword-scoped holds
- `release_litigation_hold()` — removes hold from all documents

### Production Sets
- Creates Bates-numbered production packages
- Validates no privileged documents are produced without review
- `finalize_production()` — stamps produced_at timestamp

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/dsc/health | Health check |
| GET | /api/legal/dsc/documents | List documents |
| GET | /api/legal/dsc/documents/{id} | Get document |
| POST | /api/legal/dsc/documents | Upload document |
| PUT | /api/legal/dsc/documents/{id} | Update document |
| DELETE | /api/legal/dsc/documents/{id} | Archive document |
| GET | /api/legal/dsc/documents/search | Search documents |
| GET | /api/legal/dsc/documents/{id}/family | Get document family |
| POST | /api/legal/dsc/documents/{id}/attach | Attach child document |
| POST | /api/legal/dsc/documents/{id}/redactions | Add redaction |
| GET | /api/legal/dsc/documents/{id}/redactions | List redactions |
| POST | /api/legal/dsc/documents/{id}/verify | Verify integrity |
| POST | /api/legal/dsc/documents/{id}/code | Code document |
| POST | /api/legal/dsc/documents/{id}/propagate | Propagate coding |
| POST | /api/legal/dsc/documents/{id}/retention | Set retention policy |
| GET | /api/legal/dsc/documents/destruction-eligible | List destruction eligible |
| POST | /api/legal/dsc/privilege-log | Log privilege |
| GET | /api/legal/dsc/privilege-log | List privilege log |
| POST | /api/legal/dsc/privilege-challenges | Raise privilege challenge |
| POST | /api/legal/dsc/privilege-challenges/{id}/respond | Respond to challenge |
| POST | /api/legal/dsc/privilege-challenges/{id}/rule | Rule on challenge |
| GET | /api/legal/dsc/holds | List litigation holds |
| GET | /api/legal/dsc/holds/{id} | Get hold |
| POST | /api/legal/dsc/holds | Issue litigation hold |
| POST | /api/legal/dsc/holds/{id}/release | Release hold |
| POST | /api/legal/dsc/holds/{id}/acknowledge/request | Request acknowledgement |
| POST | /api/legal/dsc/holds/{id}/acknowledge/record | Record acknowledgement |
| GET | /api/legal/dsc/holds/unacknowledged | List unacknowledged holds |
| DELETE | /api/legal/dsc/holds/{id} | Delete hold |
| GET | /api/legal/dsc/productions | List production sets |
| POST | /api/legal/dsc/productions | Create production set |
| POST | /api/legal/dsc/productions/{id}/finalize | Finalize production |
| GET | /api/legal/dsc/productions/bates-range/{matter_id} | Get Bates range |
| POST | /api/legal/dsc/deadlines | Create discovery deadline |
| GET | /api/legal/dsc/deadlines/overdue | List overdue deadlines |
| POST | /api/legal/dsc/deadlines/{id}/complete | Complete deadline |
| POST | /api/legal/dsc/costs | Record matter cost |
| GET | /api/legal/dsc/costs/summary/{matter_id} | Matter cost summary |
| GET | /api/legal/dsc/stats | Repository statistics |
| GET | /api/legal/dsc/audit | Audit events |

## Quick Usage Examples

### 1. Issue Hold + Request Acknowledgements

```python
svc = DocumentEDiscoveryService(tenant_id="acme")

# Issue hold
hold = await svc.create_litigation_hold(
    tenant_id="acme",
    matter_id="mat-002",
    title="Smith v Jones — Preservation Hold",
    description="All communications from 2024-01-01",
    custodian_ids=["emp-001", "emp-007"],
    issued_by_id="atty-003",
)

# Request acknowledgement from each custodian
for custodian_id in hold["custodian_ids"]:
    await svc.request_hold_acknowledgement(
        tenant_id="acme",
        hold_id=hold["id"],
        custodian_id=custodian_id,
        due_in_days=5,
    )

# List overdue (run daily)
overdue = await svc.list_unacknowledged_holds(tenant_id="acme")
```

### 2. Rolling Production with Bates Continuity

```python
# First production — Bates starts at 1
prod1 = await svc.create_production_set(
    tenant_id="acme",
    matter_id="mat-002",
    title="First Production",
    document_ids=["doc-001", "doc-002"],
    production_format="pdf",
    bates_prefix="ACME-",
    requesting_party="Claimant's Counsel",
    prepared_by_id="atty-003",
)
# prod1["bates_start"] == 1, prod1["bates_end"] == 2

# Second production — Bates continues from 3
prod2 = await svc.create_production_set(
    tenant_id="acme",
    matter_id="mat-002",
    title="Second Production",
    document_ids=["doc-005"],
    production_format="pdf",
    bates_prefix="ACME-",
    requesting_party="Claimant's Counsel",
    prepared_by_id="atty-003",
)
# prod2["bates_start"] == 3, prod2["bates_end"] == 3
```

### 3. FRCP Deadline Calendar + Cost Tracking

```python
from decimal import Decimal

# Create a Rule 26(a)(1) deadline
deadline = await svc.create_discovery_deadline(
    tenant_id="acme",
    matter_id="mat-002",
    deadline_type="rule26_initial",
    due_date="2026-07-15T17:00:00Z",
    description="Initial Disclosures — FRCP Rule 26(a)(1)",
    assigned_to_id="atty-003",
)

# Check overdue daily
overdue = await svc.list_overdue_deadlines(tenant_id="acme")

# Track eDiscovery costs (Decimal only — no float)
await svc.record_cost(
    tenant_id="acme",
    matter_id="mat-002",
    cost_type="processing",
    amount=Decimal("4250.00"),
    vendor="Lit Support Co",
    description="Processing 50GB native files",
)

summary = await svc.matter_cost_summary(tenant_id="acme", matter_id="mat-002")
# summary["grand_total"] == "4250.00" (string, never float)
```

## Integration Notes

| APG Capability | Integration |
|----------------|-------------|
| `leg_mtr` — Matter Management | `matter_id` links documents, holds, and costs to matters |
| `leg_prv` — Privilege Review | Privilege log entries fed to dedicated review workflows |
| `leg_cal` — Legal Calendar | Discovery deadlines synced to litigation calendar |
| `leg_bil` — Legal Billing | `matter_cost_summary()` feeds matter billing reconciliation |
| `intel_alerts` — Alerts | Overdue deadlines and unacknowledged holds trigger alerts |
| `compliance_gdpr` — GDPR Compliance | `list_destruction_eligible()` feeds GDPR erasure workflows |

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. AI-Powered Privilege Auto-Detection** [AI/ML]
- **I2. Custodian Hold Acknowledgement Workflow** [Compliance]
- **I3. Near-Duplicate & Email-Thread Detection** [AI/ML]
- **I4. Forensic Integrity Verification (Chain of Custody)** [Security]
- **I5. Document Review Coding with Near-Dup Propagation** [Feature]
- **I6. Redaction Engine with Audit Log** [Compliance]
- **I7. FRCP Discovery Deadline Calendar** [Compliance]
- **I8. Privilege Challenge & Dispute Tracker** [Feature]
- **I9. Rolling Bates Numbering (Incremental Productions)** [Feature]
- **I10. Document Family & Attachment Grouping** [Feature]
- **I11. Data Retention & Destruction Policy Engine** [Compliance]
- **I12. Matter-Level eDiscovery Cost Tracking (Decimal)** [Feature]
- **I13. Semantic PII / Entity Extraction** [AI/ML]
- **I14. Cross-Matter Document Deduplication & Coding Reuse** [Feature]
- **I15. Time-Limited Secure Share Links for Production Sets** [Security]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
