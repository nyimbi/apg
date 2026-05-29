# ACCS Capability Development Plan

## Current State

ACCS is already a domain-specific package with deterministic accessibility
models, service behavior, API helpers, view models, contract rules, theme
metadata, and package tests. The next improvement is to make critical-finding
review and closure governance a coherent first-class lifecycle instead of a
thin remediation status update.

## Packet 1: Critical Finding Review And Closure

Deliver a focused lifecycle packet:

- add explicit review and audit-event records;
- allow audits to capture critical findings while marking them as requiring
  review;
- add service methods to record formal reviews and close findings;
- enforce tenant ownership and resolution evidence on remediation closure;
- expose API helpers for review, closure, review listing, and audit events;
- expose view-model fields for review queues and compliance evidence;
- replace stale generated-package test naming with package contract tests;
- update package documentation and progress evidence.

## Implementation Steps

1. Extend `models.py` with `AccessibilityReview`, `AccessibilityAuditEvent`,
   and finding review/resolution fields.
2. Extend `service.py` with review/event stores, critical finding lifecycle
   semantics, tenant-safe remediation updates, `record_review()`,
   `close_finding()`, `list_reviews()`, and `list_audit_events()`.
3. Extend `api.py` with dependency-light helpers for the new service methods.
4. Extend `views.py` with review queue and compliance evidence view models.
5. Extend tests with positive audit-review-close flow and negative tenant,
   review, and resolution-evidence guardrails.
6. Update `cap_spec.md` with current behavior and proof commands.
7. Run focused package proof, implementation audit, publish-plan, and diff
   checks.

## Review Checklist

- Critical findings can be recorded from deterministic audits.
- Critical findings cannot be closed without an approved formal review.
- Findings cannot be closed without tenant match and resolution evidence.
- API helpers expose the same behavior as service methods.
- View models expose route, rule, review, remediation, and theme state.
- Tests cover the positive lifecycle and negative guardrails.
- Provider integrations remain adapter boundaries, not local dependencies.
