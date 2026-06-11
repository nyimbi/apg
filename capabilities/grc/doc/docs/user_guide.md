# Document Control — User Guide

**Capability**: `grc_doc` | **Version**: 2.2.0
**Author**: Nyimbi Odero | **Copyright**: © 2025 Datacraft
**Contact**: nyimbi@gmail.com | www.datacraft.co.ke

---

## Overview

`grc_doc` is the APG Document Control capability. It manages the full
lifecycle of governed documents: drafting, versioning, peer review, approval,
publication, retention enforcement, access control, digital signatures,
watermarking, lineage tracking, and AI-agent review.

Every operation is tenant-scoped. No cross-tenant data can leak because tenant
isolation is enforced at the Python layer (and, in production, at the
PostgreSQL Row Level Security layer as well).

---

## Concepts

| Term | Definition |
|------|------------|
| **Document** | A versioned, classified artifact with content, owner, type, and lifecycle status. |
| **Template** | A reusable body used to pre-populate document content during creation. |
| **Revision** | A controlled change to an existing document, incrementing its version number. |
| **Approval** | A recorded authorization by an independent approver (segregation of duties enforced). |
| **Publication** | Promoting an approved document to `published` status for distribution. |
| **Retention Policy** | A rule binding a document to a minimum retention period and legal-hold flag. |
| **Access Grant** | A time-limited, permission-scoped grant from a document to a principal. |
| **Processing Job** | Async task (OCR, classification, extraction, redaction) executed via Bytewax. |
| **Document Agent** | An AI agent (Codex, Claude Code, Pi, OpenCode) registered to review documents. |
| **Signature** | A digital-signature attestation recording signer identity and document version. |
| **Watermark** | A traceable derivative document embedding tenant + recipient + timestamp. |
| **Lineage** | Directed graph of relationships between documents (supersedes, references, watermarked copy). |

---

## Document Lifecycle

```
draft → in_review → approved → published → [archived | disposed]
                 ↑                      ↑
           (new revision)         (new revision)
```

1. **Draft** — created, may have content or template reference.
2. **In Review** — triggered automatically when a new revision is made to a
   published document.
3. **Approved** — an independent approver has signed off; restricted documents
   require a separate approver from the owner.
4. **Published** — visible to authorized consumers; triggers distribution and
   notification.
5. **Archived** — read-only; requires no active legal hold.
6. **Disposed** — destroyed, transferred, or preserved per retention policy.

---

## Quick Start

### Install / import

```python
from capabilities.grc.doc import GrcDocService
```

### Create a template

```python
svc = GrcDocService(tenant_id="acme")

template = svc.register_template(
	"tpl-isms-policy",
	"acme",
	"ISMS Policy Template",
	"1. Purpose\n2. Scope\n3. Responsibilities",
	"template-owner",
	classification="confidential",
)
```

### Create a document

```python
doc = svc.create_document(
	"doc-isms-001",
	"acme",
	"Information Security Policy",
	"policy-owner",
	content=None,                # content populated from template
	document_type="policy",
	classification="confidential",
	template_id=template["id"],
	reviewed_by="reviewer-1",
)
```

### Revise → approve → publish

```python
svc.create_revision(
	"rev-isms-001",
	"acme",
	doc["id"],
	"editor-1",
	"Updated scope section to include cloud workloads.",
	"Scope expanded per ISO 27001:2022 A.5.1 update",
	reviewed_by="reviewer-2",
)

svc.approve_document(doc["id"], "acme", "approver-1", "Reviewed and approved")
svc.publish_document(doc["id"], "acme", "publisher-1")
```

### Distribute

```python
svc.document_distribute(doc["id"], "acme", ["alice", "bob", "carol"], "publisher-1")
```

---

## Async Usage

All primary lifecycle methods have async counterparts prefixed with `async_`.
Use these in FastAPI handlers, asyncio scripts, or any context where blocking
the event loop is unacceptable.

```python
import asyncio
from capabilities.grc.doc import GrcDocService

async def run():
	svc = GrcDocService(tenant_id="acme")

	doc = await svc.async_create_document(
		"doc-async-001", "acme", "Async Policy",
		"owner-1", "Content here.", "policy", "internal",
	)
	await svc.async_approve_document(doc["id"], "acme", "approver-1", "Approved")
	await svc.async_publish_document(doc["id"], "acme", "publisher-1")

asyncio.run(run())
```

### Bulk archive (concurrent)

```python
result = await svc.async_bulk_archive(
	["doc-001", "doc-002", "doc-003"], "acme", "archivist-1"
)
# result = {"archived": 3, "failed": 0, "failures": [], "archived_at": "..."}
```

---

## Access Control

Grant a principal time-limited access to a document:

```python
svc.grant_access(
	"grant-001",
	"acme",
	doc["id"],
	"contractor-1",
	"view",
	expires_on="2026-12-31",
)
```

Supported permissions: `view`, `comment`, `edit`, `approve`, `publish`,
`archive`, `admin`.

Restricted documents always require an `expires_on` date.

---

## Retention and Legal Hold

```python
# Assign a 7-year retention policy
svc.assign_retention_policy("pol-001", "acme", doc["id"], retention_days=2555)

# Place on legal hold (blocks archive and dispose)
svc.assign_retention_policy("pol-002", "acme", doc["id"], retention_days=2555, legal_hold=True)

# Enforce retention — returns flagged documents past their retention window
result = svc.retention_enforce("acme")
# {"flagged_count": N, "flagged_documents": [...], "checked_at": "..."}

# Async enforcement with automatic event emission per flagged document
result = await svc.async_enforce_retention("acme")
```

### Disposition

```python
# Destroy an expired document (legal hold must be False)
svc.disposition_execute(doc["id"], "acme", "destroy", "records-manager-1")
```

Supported dispositions: `destroy`, `transfer`, `preserve`.

---

## Digital Signatures

```python
sig = await svc.async_sign_document(
	doc["id"],
	"acme",
	"approver-1",
	signature_metadata={"reason": "Final approval", "location": "Nairobi"},
)
# sig = {"id": "sig-...", "signer_id": "approver-1", "signed_at": "...", ...}
```

A production deployment should bind a signing adapter (HSM, AWS KMS, Azure Key
Vault) in the `async_sign_document` body. The method structure is designed for
that injection.

---

## Watermarking

Produce a traceable derivative for external distribution:

```python
wm_doc = await svc.async_watermark_document(
	doc["id"],
	"acme",
	"external-auditor",
	watermark_text="DRAFT — FOR AUDIT REVIEW ONLY — acme — 2026-06-11",
)
# wm_doc is a new document record with classification preserved and a
# "watermarked_copy_of" lineage edge back to the source.
```

---

## Document Lineage

```python
lineage = await svc.async_document_lineage(doc["id"], "acme", depth=3)
# {
#   "document_id": "doc-isms-001",
#   "upstream": ["doc-framework-001"],
#   "downstream": ["doc-proc-001", "doc-wm-001"],
#   "link_count": 5,
# }
```

Create explicit links between documents:

```python
svc.document_link(
	"doc-framework-001",
	doc["id"],
	"acme",
	link_type="references",
)
```

---

## Processing Jobs

Register a Bytewax-backed processing job:

```python
job = svc.register_processing_job(
	"job-ocr-001", "acme", doc["id"], "ocr", processor="bytewax"
)
# job["status"] = "queued"

# Mark complete with results
svc.complete_processing_job(job["id"], "acme", result={"page_count": 12, "word_count": 4800})
```

Supported job types: `classification`, `extraction`, `retention_review`,
`policy_mapping`, `quality_review`, `ocr`, `redaction`, `translation`,
`signature_verification`.

---

## Document Agents

Register an AI agent to review documents:

```python
agent = svc.register_doc_agent(
	"acme",
	"Policy Reviewer Agent",
	runtime="claude_code",
	role="policy_reviewer",
	scope="policy",
)

# Validate an action before execution
result = svc.validate_doc_agent_action(
	"acme",
	agent["id"],
	"review_document",
	privileged_scope=False,
	human_approval_recorded=True,
)
```

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.

Privileged-scope actions always require `human_approval_recorded=True`.

---

## Compliance Reporting

```python
# Synchronous
report = svc.compliance_report_doc("acme")

# Asynchronous (includes async_generated flag)
report = await svc.async_compliance_report("acme")

# Sample output:
# {
#   "total_documents": 42,
#   "approved": 18,
#   "published": 20,
#   "on_legal_hold": 3,
#   "compliance_rate_pct": 90.5,
# }
```

---

## SLA Monitoring

```python
# Record operation latency
await svc.async_record_operation_metric("publish_document", 87.3, "acme")
await svc.async_record_operation_metric("approve_document", 124.1, "acme")

# Generate SLA report
report = await svc.async_sla_report("acme", period_days=30)
# {
#   "p50_ms": 87.3, "p95_ms": 124.1, "p99_ms": 124.1,
#   "approved_document_count": 18,
#   "published_document_count": 20,
# }
```

---

## Dashboard

```python
summary = svc.dashboard_summary("acme")
# {
#   "document_count": 42, "draft_count": 4, "review_count": 2,
#   "published_count": 20, "template_count": 8, "revision_count": 67,
#   "processing_job_count": 12, "doc_agent_count": 3,
#   "overall_status": "review_required" | "operating",
# }
```

---

## Search

```python
results = svc.document_search(
	"acme",
	query="access control",
	document_type="policy",
	classification="confidential",
)
```

Use `async_document_search` in async contexts.

---

## Checkout / Checkin (Exclusive Editing)

```python
# Lock for exclusive editing
svc.document_checkout(doc["id"], "acme", "editor-1")

# Release lock after editing
svc.document_checkin(doc["id"], "acme", "editor-1")
```

---

## Collaboration

Open a document for concurrent multi-author editing:

```python
doc = svc.collaboration_draft(
	doc["id"],
	"acme",
	collaborators=["alice", "bob"],
	owner_id="owner-1",
)
# Each collaborator receives an "edit" access grant automatically.
```

---

## Audit Trail

All state-changing operations emit structured events to the `_audit_events`
list (and in production to the Bytewax `apg.grc.doc.lifecycle` event stream).

```python
events = svc.audit_events("acme")
# [{"event_type": "document_created", "record_id": "...", "emitted_at": "..."}, ...]
```

---

## Guardrails Summary

| Rule | Enforced by |
|------|------------|
| Tenant context required | `_tenant()` — raises `PermissionError` |
| Document type must be supported | capability contract rule engine |
| Classification must be supported | capability contract rule engine |
| Restricted docs require review evidence | capability contract rule engine |
| Approver must differ from owner (restricted) | capability contract rule engine |
| Publication requires prior approval | capability contract rule engine |
| Legal hold blocks archive + dispose | `archive_document`, `disposition_execute` |
| Retention minimum 365 days | capability contract rule engine |
| Restricted access grants require expiry | capability contract rule engine |
| Processing jobs require Bytewax processor | capability contract rule engine |
| Privileged agent actions require human approval | `validate_doc_agent_action` |

---

## Integration Boundary

The service makes no live external calls by default. Production adapters:

| Concern | Adapter hook |
|---------|-------------|
| Durable storage | `DocumentStore` / `AsyncPGDocumentStore` |
| Full-text search | `SearchAdapter` (Tantivy / Meilisearch) |
| Object storage | S3 / MinIO adapter for file_path binding |
| Identity + ABAC | Auth adapter injected at `grant_access` |
| Signing / KMS | Signing adapter in `async_sign_document` |
| Event streaming | Bytewax topology bound to `_emit` |
| Notifications | Outbox worker draining `_audit_events` |
| AI processing | Ollama adapter in `ml_document_classify` |

---

## Running Tests

```bash
# All capability tests
./.venv/bin/pytest -vxs capabilities/grc/doc/tests/

# Contract tests only
./.venv/bin/pytest -q capabilities/grc/doc/tests/test_package_contract.py
```
