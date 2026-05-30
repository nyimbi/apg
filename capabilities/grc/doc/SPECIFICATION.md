# Document Management Specification

## Purpose

`grc_doc` gives APG applications a governed document capability for policies,
procedures, evidence, records, contracts, reports, templates, revisions,
approvals, publication, retention, access control, processing jobs, and
AI-agent review. The package must run without external services and make every
production integration boundary explicit.

## Capability Identity

- Capability id: `grc_doc`
- Display name: `Document Management`
- Version: `2.1.0`
- Target: `python`
- Profile: `capability`
- Event stream: `apg.grc.doc.lifecycle`
- Stream processor: `bytewax`
- Theme: `grc_doc_control`

## Domain Records

### Document

Fields:

- `id`
- `tenant_id`
- `title`
- `owner_id`
- `content`
- `document_type`
- `classification`
- `template_id`
- `version`
- `reviewed_by`
- `approved_by`
- `published_by`
- `legal_hold`
- `metadata`
- `status`
- `created_at`
- `updated_at`

Supported document types are policy, procedure, evidence, contract, report,
record, and template. Supported classifications are public, internal,
confidential, and restricted.

### Template

Fields:

- `id`
- `tenant_id`
- `name`
- `body`
- `owner_id`
- `classification`
- `status`
- `created_at`

### Revision

Fields:

- `id`
- `tenant_id`
- `document_id`
- `editor_id`
- `version`
- `change_summary`
- `reviewed_by`
- `status`
- `created_at`

### Retention Policy

Fields:

- `id`
- `tenant_id`
- `document_id`
- `retention_days`
- `legal_hold`
- `status`
- `created_at`

### Access Grant

Fields:

- `id`
- `tenant_id`
- `document_id`
- `principal_id`
- `permission`
- `expires_on`
- `status`
- `created_at`

Supported permissions are view, comment, edit, approve, and admin.

### Processing Job

Fields:

- `id`
- `tenant_id`
- `document_id`
- `job_type`
- `processor`
- `result`
- `status`
- `created_at`
- `completed_at`

Supported jobs are classification, extraction, retention review, policy
mapping, and quality review. Processing metadata must use Bytewax.

### Document Agent

Fields:

- `id`
- `tenant_id`
- `name`
- `runtime`
- `role`
- `scope`
- `status`
- `created_at`

Supported runtimes are Codex, Claude Code, OpenCode, and Pi. Supported roles are
document reviewer, classification reviewer, retention reviewer, evidence
reviewer, policy reviewer, and publication reviewer.

## Lifecycle Workflows

### Repository And Template Management

1. Register document templates.
2. Create documents using content or templates.
3. Validate supported type and classification.
4. Require review evidence for restricted documents.

### Revision And Publication

1. Create revisions with editor and change summary.
2. Require review evidence for published-document revisions.
3. Approve documents with approver and note.
4. Enforce segregation of duties for restricted content.
5. Publish only approved documents.

### Retention And Access

1. Assign retention policies of at least 365 days.
2. Apply legal hold where required.
3. Block archive when legal hold is active.
4. Grant access with supported permissions.
5. Require expiry for restricted-document grants.

### Processing And Agents

1. Register supported processing jobs against documents.
2. Require Bytewax processor metadata for document processing.
3. Complete jobs with structured result payloads.
4. Register document agents by supported runtime and role.
5. Require human approval for privileged agent actions.

## Rule Engine

The deterministic rule engine returns:

- `decision`: allow, deny, or require_review;
- `matched_rules`: ordered matching rule names;
- `effects`: rule effects with reason and required action.

Rules cover tenant context, policy attachment, document fields, classification,
template completeness, revision workflow, approval, publication, retention,
legal hold, access grants, processing jobs, Bytewax routing, agent runtime and
role support, and privileged-agent approval.

## UI Contract

Routes:

- `/grc-doc/dashboard`
- `/grc-doc/documents`
- `/grc-doc/templates`
- `/grc-doc/reviews`
- `/grc-doc/retention`
- `/grc-doc/access`
- `/grc-doc/processing`
- `/grc-doc/agents`
- `/grc-doc/settings`

Screen models are framework-neutral dictionaries that generated applications
can render through the selected APG Python UI target.

## Events

Lifecycle events include:

- document created;
- template registered;
- document revised;
- document approved;
- document published;
- retention policy assigned;
- document access granted;
- processing job registered;
- processing job completed;
- document agent registered.

Each event records tenant, event type, record id, record type, status, stream,
processor, and timestamp.

## Production Adapters

The package keeps these concerns behind adapters:

- authorization;
- audit vault;
- notification;
- encrypted storage;
- search and indexing;
- workflow orchestration;
- policy management;
- durable Bytewax topology;
- theme application;
- AI-agent runtime orchestration.

## Acceptance Criteria

- Contract shape validates through APG capability tooling.
- Service executes the full document lifecycle in memory.
- Guardrails reject unsafe, incomplete, unsupported, or cross-tenant actions.
- UI routes and screen models cover all primary records.
- Semantic model includes provides, requires, rules, theme, screens, agents, and
  Bytewax streaming metadata.
- Package self-test passes.
- APG inspect, publish-plan, and implementation-audit pass for this capability.
