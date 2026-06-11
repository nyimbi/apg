# Document Control Capability — grc_doc

`grc_doc` is the APG capability packet for controlled document repositories:
creation, versioning, multi-stage approval, publication, retention enforcement,
access control, digital signatures, watermarking, lineage tracking, and
AI-agent review. The package stays dependency-light so generated APG
applications compose it immediately while production deployments attach durable
storage, search, workflow, encryption, notification, and Bytewax topology
through adapters.

**Capability**: `grc_doc` | **Version**: 2.2.0 | **Domain**: GRC
**Author**: Nyimbi Odero | **Copyright**: © 2025 Datacraft

## What It Provides

**Core document lifecycle**
- Tenant-scoped document creation with type, classification, owner, content,
  template, review evidence, and metadata.
- Template registration for repeatable policies, procedures, evidence records,
  contracts, reports, and forms.
- Controlled revision with version increments and mandatory review for published
  documents.
- Multi-stage approval and publication with segregation-of-duties guardrails.
- Retention policy assignment with minimum retention windows and legal-hold.
- Document access grants with permission scoping and expiry for restricted docs.

**Advanced controls (async methods — 17 total)**
- `async_sign_document` — digital-signature attestation; ready for HSM/KMS adapter.
- `async_watermark_document` — traceable derivative with lineage edge.
- `async_document_lineage` — upstream/downstream adjacency list for impact analysis.
- `async_enforce_retention` — background retention sweep with per-doc event emission.
- `async_bulk_archive` — concurrent archival via asyncio.gather with failure isolation.
- `async_compliance_report`, `async_disposition_execute` — awaitable pipeline ops.
- `async_record_operation_metric` / `async_sla_report` — P50/P95/P99 SLA monitoring.
- `async_create_document`, `async_approve_document`, `async_publish_document`,
  `async_create_revision`, `async_document_search`, `async_grant_access`,
  `async_dashboard_summary`, `async_register_processing_job` — async lifecycle wrappers.

**Processing and agents**
- Bytewax-backed processing jobs: classification, extraction, OCR, redaction, translation.
- Document-agent registration for Codex, Claude Code, OpenCode, and Pi review teams.
- APG UI route metadata, screen models, semantic metadata, and release evidence.

## Package Layout

| File | Purpose |
|------|---------|
| `SPECIFICATION.md` | Records, workflows, rules, UI, events, adapter boundaries |
| `PLAN.md` | Implementation and review plan |
| `WORLD_CLASS_IMPROVEMENTS.md` | 15 targeted improvements roadmap |
| `cap_spec.md` | Executable runtime contract summary |
| `capability_contract.py` | APG contract and deterministic rule engine |
| `service.py` | Lifecycle service — sync + 17 async methods |
| `models.py` | Pydantic v2 and SQLAlchemy models |
| `api.py` | Composition helpers |
| `views.py` | Screen models (no framework imports) |
| `app.py` | Semantic model, component manifest, self-test |
| `domain/` | Domain events, adapter interfaces, business rules |
| `database/` | PostgreSQL schema and store adapter |
| `docs/user_guide.md` | End-user and operator guide |
| `tests/` | Contract, lifecycle, guardrail, API, composition tests |

## Runtime Lifecycle

1. Register templates for repeatable policy or evidence structures.
2. Create documents from content or templates.
3. Require review for restricted documents.
4. Create revisions as controlled versioned changes.
5. Approve documents with independent approvers (restricted requires separate approver).
6. Publish approved documents.
7. Assign retention policies and legal holds.
8. Grant document access with supported permissions and expiry where required.
9. Register Bytewax processing jobs and complete them with structured results.
10. Register document agents that review, prepare, and recommend within human-approval bounds.

## Usage

### Synchronous (CLI / test) path

```python
from capabilities.grc.doc import GrcDocService

service = GrcDocService()

template = service.register_template(
	"tpl-access-policy",
	"tenant-a",
	"Access Policy Template",
	"Policy body",
	"template-owner",
)
document = service.create_document(
	"doc-access-policy",
	"tenant-a",
	"Access Policy",
	"document-owner",
	"Initial access policy",
	"policy",
	"confidential",
	template["id"],
)
service.create_revision(
	"rev-access-policy",
	"tenant-a",
	document["id"],
	"editor",
	"Updated access policy",
	"Clarified quarterly access review",
)
service.approve_document(document["id"], "tenant-a", "approver", "Ready")
service.publish_document(document["id"], "tenant-a", "publisher")
print(service.dashboard_summary("tenant-a"))
```

### Async path (FastAPI / asyncio apps)

```python
import asyncio
from capabilities.grc.doc import GrcDocService

async def main():
	svc = GrcDocService(tenant_id="tenant-a")
	doc = await svc.async_create_document(
		"doc-001", "tenant-a", "Security Policy",
		"owner-1", "Content...", "policy", "confidential",
	)
	await svc.async_approve_document(doc["id"], "tenant-a", "approver-1", "LGTM")
	await svc.async_publish_document(doc["id"], "tenant-a", "publisher-1")
	sig = await svc.async_sign_document(doc["id"], "tenant-a", "signer-1")
	wm = await svc.async_watermark_document(doc["id"], "tenant-a", "external-auditor")
	lineage = await svc.async_document_lineage(doc["id"], "tenant-a")
	await svc.async_record_operation_metric("publish_document", 42.5, "tenant-a")
	print(await svc.async_sla_report("tenant-a"))

asyncio.run(main())
```

### Composition via `api.py`

```python
from capabilities.grc.doc import api

status = api.capability_status("tenant-a")
records = api.list_records("documents", "tenant-a")
```

## Guardrails

- Tenant context is required on all operations.
- Documents require title, owner, supported type, supported classification, content or template.
- Restricted documents require review evidence.
- Templates require name, body, owner, and supported classification.
- Revisions require document, editor, and change summary.
- Published document revisions require review evidence.
- Approvals require document, approver, note, and segregation of duties for restricted content.
- Publication requires approval and publisher.
- Retention requires at least 365 days.
- Legal hold blocks archive and dispose.
- Access grants require expiry for restricted documents.
- Processing jobs require document, supported type, and Bytewax processor.
- Privileged document-agent actions require recorded human approval.

## Integration Boundary

No live external connections by default. Production adapters:

- Identity, authorization, and access policy
- Audit vault and immutable event storage
- Encrypted document/object storage
- Full-text search (Tantivy / Meilisearch)
- Workflow and approval routing
- Notifications and collaboration
- Policy management and obligation mapping
- Durable Bytewax topology and event sinks
- AI-agent runtime orchestration (Ollama-hosted models)

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/grc/doc/__init__.py \
  capabilities/grc/doc/capability_contract.py \
  capabilities/grc/doc/service.py \
  capabilities/grc/doc/api.py capabilities/grc/doc/views.py \
  capabilities/grc/doc/app.py \
  capabilities/grc/doc/tests/test_package_contract.py

./.venv/bin/pytest -q capabilities/grc/doc/tests/test_package_contract.py
./.venv/bin/python capabilities/grc/doc/app.py
./.venv/bin/apg capabilities inspect grc_doc --json
./.venv/bin/apg capabilities publish-plan capabilities/grc/doc --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/grc/doc --json
```
