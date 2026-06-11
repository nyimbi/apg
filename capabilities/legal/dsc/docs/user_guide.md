# Document & eDiscovery (leg_dsc) — User Guide

## Overview

`leg_dsc` is a legal-grade document repository providing version control, attorney-client privilege logging, litigation hold management, eDiscovery production with rolling Bates numbering, review coding with near-duplicate propagation, a redaction engine, FRCP discovery deadline calendar, privilege challenge tracking, matter cost tracking, and hold acknowledgement workflows.

All monetary values use `Decimal`, never `float`. All operations are fully async.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Document | Versioned, metadata-rich file with forensic hash and family linkage |
| Privilege Log | Formal record of attorney-client and work-product assertions |
| Privilege Challenge | Opposing counsel challenge to a privilege assertion, tracked through ruling |
| Litigation Hold | Freeze on document modification/deletion for preservation |
| Hold Acknowledgement | Custodian's formal receipt of a hold order — court-admissible |
| Production Set | Bates-numbered document package for disclosure to opposing counsel |
| Rolling Bates | Matter-scoped counter ensuring no Bates gaps or duplicates across productions |
| Review Coding | Responsive/non_responsive/redact/withhold decision recorded per document |
| Near-Dup Propagation | Auto-applying a coding decision to the near-duplicate cluster |
| Redaction | Page-level, bbox-scoped content masking with reason and redactor recorded |
| FRCP Deadline | Rule 26/34/45 deadline tracked with days_overdue computation |
| Retention Policy | Document-level destroy_after_date — skipped if document is on hold |
| Matter Cost Entry | Decimal cost entry (processing/hosting/review/production) per matter |

---

## Instantiation

```python
from capabilities.legal.dsc.service import DocumentEDiscoveryService

svc = DocumentEDiscoveryService(tenant_id="acme")
```

Multi-tenant: pass `tenant_id` explicitly on every call. The instance-level `tenant_id` is only used as a fallback default.

---

## Document Management

### Upload a Document

```python
doc = await svc.create_document(
    tenant_id="acme",
    title="Board Minutes 2026-01-15",
    document_type="internal",          # must be in DOCUMENT_TYPES
    owner_id="atty-003",
    file_reference="s3://legal-docs/board-minutes-2026-01-15.pdf",
    matter_id="mat-002",
    file_size_bytes=142_000,
    mime_type="application/pdf",
    content_sha256="e3b0c44298fc1c149afb...",  # optional forensic hash
    is_privileged=False,
)
```

Supported `document_type` values: `pleading`, `brief`, `contract`, `correspondence`, `evidence`, `internal`, `court_order`, `affidavit`, `exhibit`, `expert_report`.

### Forensic Integrity Verification (I4)

```python
result = await svc.verify_integrity(
    tenant_id="acme",
    document_id="doc-abc123",
    current_sha256="e3b0c44298fc1c149afb...",
)
# result["verified"] == True|False
# result["status"] == "verified" | "tampered" | "no_hash_stored"
```

Use this before every production to confirm no tampering since ingest.

### Document Families / Attachments (I10)

Email attachments must travel with their parent email in every production (FRCP Rule 34(b)(2)(E)).

```python
# Attach a child document to a parent
link = await svc.attach_document(
    tenant_id="acme",
    child_doc_id="doc-attach-001",
    parent_doc_id="doc-email-001",
)

# Retrieve full family tree (parent-first)
family = await svc.get_document_family(tenant_id="acme", document_id="doc-email-001")
```

All family members share a `family_id`. Production sets should retrieve families using `get_document_family()` to ensure complete disclosure.

---

## Review Coding (I5)

### Code a Document

```python
coding = await svc.code_document(
    tenant_id="acme",
    document_id="doc-abc123",
    coding="responsive",        # responsive | non_responsive | needs_review | redact | withhold
    reviewer_id="rev-001",
    note="Directly responsive to RFP #7",
)
```

### Propagate to Near-Duplicates

```python
result = await svc.propagate_coding(tenant_id="acme", document_id="doc-abc123")
# result["propagated_to"] == ["doc-xyz456", "doc-pqr789"]
# Near-dups receive same coding at confidence=0.85 (flagged for optional re-review)
```

Near-duplicate cluster membership (`near_dup_cluster_id`) is populated by a separate deduplication scan and stored on the document record.

---

## Redaction Engine (I6)

### Add a Redaction

```python
rdx = await svc.add_redaction(
    tenant_id="acme",
    document_id="doc-abc123",
    page=3,
    bbox=[100.0, 200.0, 400.0, 220.0],  # [x1, y1, x2, y2] in points
    reason="attorney_client_privilege",
    redacted_by="atty-003",
)
```

### List Redactions (Privilege Log Export)

```python
redactions = await svc.list_redactions(tenant_id="acme", document_id="doc-abc123")
# Returns list of {id, page, bbox, reason, redacted_by, created_at}
```

Export this list as the redaction manifest accompanying any privilege log submission.

---

## Privilege Log

### Log Privilege Manually

```python
entry = await svc.log_privilege(
    tenant_id="acme",
    document_id="doc-abc123",
    privilege_type="attorney_client",   # see PRIVILEGE_TYPES
    basis="Confidential legal advice requested by CEO re: acquisition",
    logged_by_id="atty-003",
    notes="Email chain between CEO and outside counsel",
)
```

Privilege types: `attorney_client`, `work_product`, `common_interest`, `settlement`, `deliberative`.

### Raise a Privilege Challenge (I8)

When opposing counsel challenges a privilege assertion:

```python
challenge = await svc.raise_privilege_challenge(
    tenant_id="acme",
    privilege_id="prv-abc123",
    challenger_id="opp-counsel-001",
    basis="Document predates any litigation; no dominant purpose of legal advice",
)
```

### Respond to a Challenge

```python
response = await svc.respond_to_challenge(
    tenant_id="acme",
    challenge_id="chg-xyz456",
    response_text="Document was created at counsel's direction in anticipation of...",
    supporting_doc_ids=["doc-engagement-letter"],
)
```

### Record Court Ruling

```python
ruling = await svc.rule_on_challenge(
    tenant_id="acme",
    challenge_id="chg-xyz456",
    ruling="upheld",    # "upheld" | "overruled"
    ruled_by="judge-001",
)
# If "overruled": document.is_privileged set to False automatically
```

---

## Litigation Holds

### Issue a Hold

```python
hold = await svc.create_litigation_hold(
    tenant_id="acme",
    matter_id="mat-002",
    title="Smith v Jones — Preservation Hold",
    description="All communications from 2024-01-01 to present",
    custodian_ids=["emp-001", "emp-007", "emp-012"],
    issued_by_id="atty-003",
    scope_query="smith jones contract",   # optional keyword scope
)
```

Hold auto-applies to all matching active documents. Documents under hold cannot be modified or deleted.

### Hold Acknowledgement Workflow (I2)

```python
# Step 1: request acknowledgements from all custodians
for custodian_id in hold["custodian_ids"]:
    await svc.request_hold_acknowledgement(
        tenant_id="acme",
        hold_id=hold["id"],
        custodian_id=custodian_id,
        due_in_days=5,
    )

# Step 2: record receipt when custodian signs
await svc.record_acknowledgement(
    tenant_id="acme",
    hold_id=hold["id"],
    custodian_id="emp-001",
    signature_reference="docusign-envelope-98765",
)

# Step 3: daily escalation check
overdue_acks = await svc.list_unacknowledged_holds(tenant_id="acme")
# Each item includes days_overdue; escalate to partner for follow-up
```

### Release a Hold

```python
released = await svc.release_litigation_hold(
    tenant_id="acme",
    hold_id=hold["id"],
    released_by="atty-003",
)
```

---

## FRCP Discovery Deadline Calendar (I7)

### Create a Deadline

```python
ddl = await svc.create_discovery_deadline(
    tenant_id="acme",
    matter_id="mat-002",
    deadline_type="rule26_initial",     # see DEADLINE_TYPES
    due_date="2026-07-15T17:00:00Z",
    description="Initial Disclosures — FRCP Rule 26(a)(1)",
    assigned_to_id="atty-003",
)
```

Deadline types: `rule26_initial`, `rule26_supplemental`, `meet_and_confer`, `production_response`, `deposition_notice`, `expert_disclosure`.

### Check Overdue Deadlines (Run Daily)

```python
overdue = await svc.list_overdue_deadlines(tenant_id="acme")
for item in overdue:
    print(f"{item['deadline_type']} — {item['days_overdue']} days overdue")
```

### Complete a Deadline

```python
await svc.complete_deadline(
    tenant_id="acme",
    deadline_id=ddl["id"],
    completed_by="atty-003",
)
```

---

## Rolling Bates Numbering (I9)

Each matter maintains an independent Bates counter. Every new production set starts at `prior_end + 1`.

```python
# Check current high-water mark before production
range_info = await svc.get_bates_range(tenant_id="acme", matter_id="mat-002")
# {"current_high_water": 0, "next_start": 1}

prod1 = await svc.create_production_set(
    tenant_id="acme", matter_id="mat-002",
    title="First Production", document_ids=["doc-001", "doc-002"],
    production_format="pdf", bates_prefix="ACME-",
    requesting_party="Claimant's Counsel", prepared_by_id="atty-003",
)
# prod1["bates_start"]=1, prod1["bates_end"]=2

prod2 = await svc.create_production_set(
    tenant_id="acme", matter_id="mat-002",
    title="Second Production", document_ids=["doc-005", "doc-006", "doc-007"],
    production_format="pdf", bates_prefix="ACME-",
    requesting_party="Claimant's Counsel", prepared_by_id="atty-003",
)
# prod2["bates_start"]=3, prod2["bates_end"]=5  — no gap, no duplicate
```

---

## Retention & Destruction Policy (I11)

### Assign a Policy

```python
await svc.set_retention_policy(
    tenant_id="acme",
    document_id="doc-abc123",
    policy_id="pol-7yr-contracts",
    destroy_after_date="2033-01-01T00:00:00Z",
)
```

### List Documents Eligible for Destruction

```python
eligible = await svc.list_destruction_eligible(tenant_id="acme")
# Documents past destroy_after_date and NOT on any litigation hold
# Each item includes days_past_retention
for doc in eligible:
    print(f"Document {doc['id']} — {doc['days_past_retention']} days past retention")
```

Documents under a litigation hold are excluded from this list regardless of their retention date.

---

## Matter Cost Tracking (I12)

All amounts are `Decimal`. Never pass `float`.

```python
from decimal import Decimal

# Record costs
await svc.record_cost(
    tenant_id="acme",
    matter_id="mat-002",
    cost_type="processing",     # processing | hosting | review | production | collection | other
    amount=Decimal("4250.00"),
    vendor="Lit Support Co",
    description="Processing 50GB native files — first wave",
)

await svc.record_cost(
    tenant_id="acme",
    matter_id="mat-002",
    cost_type="review",
    amount=Decimal("18750.00"),
    vendor="Contract Review LLC",
    description="First-pass review 2500 docs @ $7.50/doc",
)

# Get summary
summary = await svc.matter_cost_summary(tenant_id="acme", matter_id="mat-002")
print(summary["grand_total"])         # "23000.00" — string, never float
print(summary["by_type"])             # {"processing": "4250.00", "review": "18750.00"}
```

---

## Repository Statistics

```python
stats = await svc.repository_stats(tenant_id="acme")
# {
#   "total_documents": 142,
#   "total_size_bytes": 2_400_000_000,
#   "by_type": {"contract": 45, "correspondence": 62, ...},
#   "privileged_count": 18,
#   "on_hold_count": 87,
#   "active_holds": 3,
#   "destruction_eligible": 12,
# }
```

---

## Audit Trail

Every service call emits an audit event. Retrieve recent events:

```python
events = await svc.get_audit_events(tenant_id="acme", limit=50)
for evt in events:
    print(f"{evt['created_at']}  {evt['event_type']}  {evt['entity_id']}")
```

---

## APG Integration Map

| Capability | Integration Point |
|------------|-------------------|
| `leg_mtr` Matter Management | All documents, holds, costs link to `matter_id` |
| `leg_cal` Legal Calendar | `create_discovery_deadline()` feeds calendar entries |
| `leg_bil` Legal Billing | `matter_cost_summary()` feeds billing reconciliation |
| `intel_alerts` Alerts | Overdue deadlines and unacknowledged holds trigger alerts |
| `compliance_gdpr` GDPR | `list_destruction_eligible()` feeds GDPR erasure workflow |
| `leg_prv` Privilege Review | Privilege log entries passed to dedicated review queue |

---

## Error Reference

| Exception | Cause |
|-----------|-------|
| `ValueError` | Invalid enum value, bad arguments, business rule violation |
| `KeyError` | Entity not found or tenant mismatch |
| `TypeError` | Non-Decimal amount passed to `record_cost()` |
| `PermissionError` | (future RBAC) — insufficient document access rights |

---

*© 2025 Datacraft — www.datacraft.co.ke*
