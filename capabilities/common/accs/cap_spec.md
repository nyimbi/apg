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

## Provided Services

- `accessibility_standard_registry`
- `accessibility_target_registry`
- `accessibility_audit_runner`
- `accessibility_findings_board`
- `accessibility_remediation_queue`
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
targets, findings, remediation tasks, and completed audit runs.
`accessibility_engine.py` performs deterministic checks for contrast, semantic
labels, keyboard navigation, and media captions. Publication validation applies
the same rule engine used by package contracts so generated APG applications can
block inaccessible UI before release.

## UI

The package exposes 8 APG Python UI route contract(s) through `views.py` and the
package semantic model. View models cover the dashboard, audit console,
findings board, remediation queue, and assistive preview.

## Theme

The package uses the `accs_accessibility_ops` APG theme contract for audit
scores, severity bands, finding boards, compliance panels, and assistive
semantic-tree previews.
