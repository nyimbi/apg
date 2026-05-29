# AGNT Capability Development Plan

## Current State

AGNT is already a domain-specific package with agent runtime registration,
agent declarations, teams, handoff validation, execution planning, API helpers,
route metadata, theme metadata, and package tests. The next improvement is to
make external runtime approval explicit so rapidly changing AI providers can be
integrated without weakening governance.

## Packet 1: External Runtime Approval Lifecycle

Deliver a focused lifecycle packet:

- add runtime approval and audit-event records;
- let tenants request approval for external runtimes;
- allow approved requests to register provider-neutral runtimes;
- block rejected or pending runtimes from direct use;
- expose API helpers for request, approval, approval listing, and audit events;
- expose view models for runtime approval queue and governance evidence;
- replace stale generated-package test naming with package contract tests;
- update package documentation and progress evidence.

## Implementation Steps

1. Extend `models.py` with `RuntimeApprovalRequest` and `AgentAuditEvent`.
2. Extend `service.py` with approval/event stores, request and decision
   methods, tenant-safe runtime approval, and summary counts.
3. Extend `api.py` with dependency-light helpers for the new lifecycle.
4. Extend `views.py` with approval queue and governance evidence models.
5. Extend tests with positive request-approve-register-agent-team-plan flow and
   negative rejected/pending/tenant/sandbox guardrails.
6. Update `cap_spec.md` with current behavior and proof commands.
7. Run focused package proof, implementation audit, publish-plan, and diff
   checks.

## Review Checklist

- External runtimes cannot be registered or used without approval.
- Approval requests preserve provider-neutral runtime metadata.
- Workspace-aware runtimes require sandbox policy before request or approval.
- API helpers expose the same behavior as service methods.
- View models expose approval state, runtime state, and audit evidence.
- Tests cover the positive lifecycle and negative guardrails.
- Provider SDKs remain adapter boundaries, not package dependencies.
