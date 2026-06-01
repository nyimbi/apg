# AUDL Capability Specification

## Identity

- Capability ID: `audl`
- Display name: Audit Logging
- Category: `common`
- Owner: APG Platform Team
- Runtime shell: `apg_python`
- Theme: `audl_forensics`

## Purpose

AUDL is the tenant-scoped audit evidence backbone for APG applications. It
captures immutable audit events, verifies checksums, governs export and purge
operations, tracks legal hold, supports investigations, registers first-class
audit agents, and exposes audit state through dependency-light API helpers and
view models.

The package must remain usable without Elasticsearch, a running Bytewax worker,
external SIEM systems, blockchain services, machine-learning providers, or
Flask/FastAPI servers. Those systems remain adapter boundaries. Local package
proof focuses on deterministic audit lifecycle governance, chain-of-custody
evidence, tenant isolation, agent composition, and composability.

## Users And Outcomes

- Application components can append audit events with tenant, actor, action,
  resource, checksum, and severity context.
- Compliance operators can place and release legal holds with human evidence.
- Security teams can open, update, and close investigations over event sets.
- Data protection reviewers can approve regulated exports only when masking is
  enabled for PII-bearing evidence.
- Retention operators can request purges, but legal hold and dual-control
  rules block unsafe deletion.
- AI agents can be registered as governed AUDL workers for evidence, export,
  purge, investigation, compliance, and legal-hold review.
- Generated APG applications can compose AUDL with AUTH, MTEN, CONF, SECU,
  NTFY, WFLO, MONI, APIG, and AICR without binding to one storage or UI stack.
- Operators can inspect durable review and denial evidence for exports, purges,
  privileged audit agents, batches, lifecycle events, and governance events
  without replaying rules.

## Domain Model

AUDL owns these package-level records:

- `AuditLifecycleEvent`: immutable tenant-scoped audit event with checksum.
- `AuditLegalHoldRecord`: hold state, scope, approver, and release evidence.
- `AuditExportRequest`: governed export request, masking decision, and review.
- `AuditPurgeRequest`: dual-control purge request and outcome.
- `AuditInvestigationRecord`: investigation lifecycle over audit event IDs.
- `AuditAgentRecord`: first-class audit agent with runtime, role, owner,
  purpose, approval gate, and configuration.
- `AuditBatchEvidence`: accepted or denied Bytewax audit batch validation
  evidence.
- `AuditGovernanceEvent`: tenant-scoped evidence event for AUDL decisions.

All mutable package-level state must be tenant-qualified so duplicate IDs in
different tenants cannot collide.

## Lifecycle

The focused lifecycle is:

1. Append an audit event with tenant, actor, action, resource, severity, and
   checksum context.
2. Reject immutable audit events whose checksum cannot be verified.
3. Open a legal hold over a tenant, resource, event, or query scope.
4. Request a regulated export over audited evidence.
5. Deny PII-bearing exports unless masking is enabled.
6. Store masked PII-bearing exports as `review_required` before approval or
   rejection.
7. Require reviewer identity and notes before export approval or rejection.
8. Request purge with requester, reviewer, reason, and scope.
9. Store purge requests as `review_required` and deny purge while legal hold is
   active or dual-control evidence is missing.
10. Open and close investigations with assigned owner and resolution evidence.
11. Register audit agents on approved runtimes and roles.
12. Store privileged audit-agent registrations without human approval as
   `pending_review`.
13. Validate audit batches before ingestion and require the Bytewax lifecycle
    stream for high-volume batch work.
14. Persist denied export, purge, and batch evidence before raising
    `PermissionError`.
15. Emit tenant-scoped governance events for every lifecycle decision.

## Rules And Guardrails

The contract rules are executable guardrails:

- `require_tenant_context`: operations require tenant context.
- `immutable_events_require_checksum`: immutable audit records require checksum
  verification.
- `legal_hold_blocks_purge`: audit data under legal hold cannot be purged.
- `regulated_exports_require_masking`: PII-bearing exports require masking.
- `critical_events_require_escalation`: critical events require escalation
  routing.
- `high_volume_ingestion_requires_stream_processing`: large batches require
  stream-processing safeguards.
- `bytewax_event_stream_required`: audit batch ingestion must use the Bytewax
  lifecycle stream.
- `audit_agent_runtime_supported`: audit agents must use one of `codex`,
  `claude_code`, `opencode`, or `pi`.
- `audit_agent_role_supported`: audit agents must use an AUDL review role.
- `audit_agent_privileged_action_requires_approval`: privileged audit-agent
  roles require human approval evidence or review.
- `regulated_export_requires_review`: masked PII-bearing exports require
  review before release.
- `audit_purge_requires_dual_control_review`: purge requests require
  dual-control review before execution.

Service methods must enforce these rules and expose the same decisions through
API helpers and view models.

Every review-required or denied lifecycle record must expose:

- `policy_decision`
- `matched_rules`
- `review_reasons`
- `audit_evidence`

## UI And Theme

AUDL exposes route and view-model surfaces for:

- dashboard summary;
- event explorer;
- live timeline;
- investigation workbench;
- legal hold console;
- export review queue;
- purge review queue;
- compliance center;
- audit agent roster;
- reporting studio;
- rule workbench and settings.

The `audl_forensics` theme must provide semantic tokens and component metadata
for audit timelines, investigation case cards, compliance scorecards, severity
badges, hold indicators, export review panels, purge approval warnings, audit
agent rosters, and Bytewax stream indicators.

## Adapter Boundaries

These integrations remain replaceable:

- Elasticsearch, OpenSearch, data lake, and immutable object storage;
- Bytewax stream processors and queue/event-bus adapters;
- SIEM, GRC, DLP, notification, and case-management exporters;
- cryptographic timestamping and blockchain proof providers;
- ML anomaly detection and natural-language search providers;
- Flask/FastAPI web servers and frontend rendering stacks.

Local package tests must not require those systems.

## Acceptance Gates

Focused AUDL proof:

```bash
./.venv/bin/pytest -q capabilities/common/audl/tests/test_capability_contract.py capabilities/common/audl/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/audl --json
./.venv/bin/apg capabilities publish-plan capabilities/common/audl --json
git diff --check -- capabilities/common/audl
```
