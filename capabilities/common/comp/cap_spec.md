# Compliance Management Capability Specification

- **Capability Name**: Compliance Management
- **Capability ID**: `comp`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package provides the executable APG compliance-management runtime for
`comp`. It owns tenant compliance frameworks, control libraries, encrypted
evidence records, control assessments, findings, report approvals,
attestations, and immutable audit-event metadata behind the APG capability
contract.

The implementation is dependency-light and deterministic so generated APG
applications can compose it without requiring a live GRC platform, regulator
feed, document repository, scanner, DLP engine, or audit-log sink. Those
integrations remain explicit APG capability boundaries.

## Provided Services

- `framework_management`
- `control_assurance`
- `evidence_collection`
- `finding_remediation`
- `regulatory_reporting`
- `attestation_management`
- `compliance_audit_events`

## Required Services

- `audl`
- `dlpd`
- `encr`
- `auth`

Optional composition targets include `secu`, `mten`, `idfd`, and `ztna`.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `control_requires_owner`
- `stale_evidence_requires_refresh`
- `regulated_data_requires_dlp`
- `report_requires_approval`
- `overdue_finding_requires_escalation`

## UI

The package exposes APG Python route contracts through `views.py` and provides
dashboard, framework matrix, control library, evidence vault, assessment
history, remediation board, report builder, attestation center, and audit
timeline view models.

## Theme

The package uses the `comp_compliance_command_center` APG theme contract.
