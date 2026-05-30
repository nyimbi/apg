# Document Management Capability

`grc_doc` is the APG capability packet for governed document repositories,
templates, revisions, approvals, publication, retention, access, processing, and
AI-agent review. It keeps the package boundary dependency-light so generated APG
applications can compose it immediately while production deployments attach
durable storage, search, workflow, security, notification, and Bytewax topology
through adapters.

## What It Provides

- Tenant-scoped document creation with document type, classification, owner,
  content, template, review evidence, and metadata.
- Template registration for repeatable policies, procedures, evidence records,
  contracts, reports, and document records.
- Controlled document revision with version increments and review requirements
  for published documents.
- Approval and publication workflows with segregation-of-duties guardrails.
- Retention policy assignment with minimum retention and legal-hold handling.
- Document access grants with supported permissions and restricted-document
  expiry requirements.
- Bytewax-backed processing job metadata for classification, extraction,
  retention review, policy mapping, and quality review.
- First-class document-agent registration for Codex, Claude Code, OpenCode, and
  Pi review teams.
- APG UI route metadata, framework-neutral screen models, compact theme tokens,
  semantic metadata, package manifest, and release evidence.

## Package Layout

- `SPECIFICATION.md` defines records, workflows, rules, UI, events, adapter
  boundaries, and acceptance criteria.
- `PLAN.md` records the implementation and review plan for this lifecycle
  packet.
- `cap_spec.md` summarizes the current executable runtime contract.
- `capability_contract.py` exposes the executable APG contract and deterministic
  rule engine.
- `service.py` implements the dependency-light lifecycle service.
- `api.py` exposes composition helpers.
- `views.py` exposes screen models without framework imports.
- `app.py` exposes semantic model, component manifest, and self-test.
- `tests/test_package_contract.py` verifies the package contract, lifecycle,
  guardrails, API, views, and app surface.

## Runtime Lifecycle

1. Register templates when teams need repeatable policy or evidence structures.
2. Create documents from content or templates.
3. Require review for restricted documents.
4. Create revisions as controlled versioned changes.
5. Approve documents with independent approvers for restricted content.
6. Publish approved documents.
7. Assign retention policies and legal holds.
8. Grant document access using supported permissions and expiry where required.
9. Register Bytewax processing jobs and complete them with structured results.
10. Register document agents that review, prepare, and recommend within explicit
    human-approval boundaries.

## Usage

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

Generated APG applications can use `api.py`:

```python
from capabilities.grc.doc import api

status = api.capability_status("tenant-a")
records = api.list_records("documents", "tenant-a")
```

## Guardrails

- Tenant context is required.
- Write operations require policy context.
- Documents require title, owner, supported type, supported classification, and
  content or template.
- Restricted documents require review evidence.
- Templates require name, body, owner, and supported classification.
- Revisions require document, editor, and change summary.
- Published document revisions require review evidence.
- Approvals require document, approver, note, and segregation of duties for
  restricted content.
- Publication requires approval and publisher.
- Retention requires a valid document and at least 365 days.
- Legal hold blocks archive.
- Access grants require document, principal, supported permission, and expiry
  for restricted documents.
- Processing jobs require document, supported job type, and Bytewax processor.
- Document batches and events require Bytewax metadata.
- Document agents must use supported runtimes and roles.
- Privileged document-agent actions require recorded human approval.

## Integration Boundary

This package does not open live external connections by default. Production
deployments should bind these concerns through adapters:

- identity, authorization, and access policy;
- audit vault and immutable event storage;
- encrypted document/object storage;
- search and indexing;
- workflow and approval routing;
- notifications and collaboration;
- policy management and obligation mapping;
- durable Bytewax topology and event sinks;
- AI-agent runtime orchestration.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/grc/doc/__init__.py capabilities/grc/doc/capability_contract.py capabilities/grc/doc/service.py capabilities/grc/doc/api.py capabilities/grc/doc/views.py capabilities/grc/doc/app.py capabilities/grc/doc/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/grc/doc/tests/test_package_contract.py
./.venv/bin/python capabilities/grc/doc/app.py
./.venv/bin/apg capabilities inspect grc_doc --json
./.venv/bin/apg capabilities publish-plan capabilities/grc/doc --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/grc/doc --json
```
