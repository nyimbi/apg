# Accessibility Services Capability Specification

- **Capability Name**: Accessibility Services
- **Capability ID**: `accs`
- **Category**: common
- **Version**: 1.0.0

## Purpose

ACCS provides first-class accessibility governance for APG applications. It
registers accessibility standards, audits UI/content/media targets, records
findings, creates remediation tasks, validates publication guardrails, and
projects operational view models for audit consoles, findings boards,
remediation queues, assistive previews, and compliance dashboards.
Critical findings now move through an explicit review-and-closure lifecycle so
audits can capture severe accessibility failures while closure remains blocked
until an approved formal review and resolution evidence are recorded.

## Provided Services

- `accessibility_standard_registry`
- `accessibility_target_registry`
- `accessibility_audit_runner`
- `accessibility_findings_board`
- `accessibility_remediation_queue`
- `accessibility_review_governance`
- `accessibility_publication_validation`
- `assistive_preview_view_models`

## Required Services

- `tenant_context`
- `them`
- `i18n`
- `nlpc`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations. The service applies WCAG 2.2 AA defaults and supports tenant-owned
standard profiles for alternate regulatory targets such as EN 301 549.

## Rules

- `tenant_context_required`
- `audit_requires_standard`
- `violation_requires_remediation_owner`
- `published_ui_requires_contrast`
- `media_requires_captions`
- `critical_issue_requires_review`

## Runtime Behavior

`service.py` owns dependency-light in-memory registries for standards, audit
targets, findings, remediation tasks, formal reviews, audit events, and
completed audit runs.
`accessibility_engine.py` performs deterministic checks for contrast, semantic
labels, keyboard navigation, and media captions. Publication validation applies
the same rule engine used by package contracts so generated APG applications can
block inaccessible UI before release.

Primary lifecycle:

1. Register an accessibility standard and tenant-owned target.
2. Run a deterministic audit.
3. Record findings and remediation tasks with evidence and owners.
4. Mark critical findings as requiring review.
5. Record a formal review decision.
6. Close the finding only after an approved review and resolution evidence
   exist.
7. Emit audit events for finding, review, remediation, and closure changes.

## UI

The package exposes 8 APG Python UI route contract(s) through `views.py` and the
package semantic model. View models cover the dashboard, audit console,
findings board, remediation queue, critical review queue, compliance evidence,
and assistive preview.

## Theme

The package uses the `accs_accessibility_ops` APG theme contract for audit
scores, severity bands, finding boards, compliance panels, and assistive
semantic-tree previews.

## Focused Verification

```bash
./.venv/bin/pytest -q capabilities/common/accs/test_capability_contract.py capabilities/common/accs/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/accs --json
./.venv/bin/apg capabilities publish-plan capabilities/common/accs --json
git diff --check -- capabilities/common/accs
```
