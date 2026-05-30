# COMP Capability Specification

## Purpose

`comp` provides a composable Compliance Management capability for APG
applications. It turns frameworks, obligations, controls, evidence,
assessments, findings, reports, attestations, and audit records into executable
compliance operations with deterministic guardrails and UI-ready view models.

The capability does not replace a live GRC suite, document repository,
regulator feed, policy engine, DLP product, identity provider, or audit-log
sink. Those systems are adapter responsibilities. The local package proves the
domain lifecycle, contract, rules, state transitions, UI payloads, and package
evidence.

## Scope

In scope:

- tenant-scoped compliance frameworks;
- obligation mappings and policy-version evidence;
- tenant-scoped controls;
- regulated-data and DLP linkage guardrails;
- encrypted immutable evidence records;
- control assessments and evidence freshness checks;
- findings, escalation, remediation, and resolution evidence;
- report preparation, independent approval, attestation, publication, and
  critical-finding blocking;
- hashed audit-event metadata;
- route, permission, view-model, theme, and adapter metadata;
- package self-test, semantic model, manifest, release report, audit, and
  publish-plan evidence.

Out of scope for the local package:

- live GRC platform synchronization;
- live document repository storage;
- live regulator submission;
- live DLP inspection;
- browser rendering;
- persistent database migrations;
- live Bytewax execution.

## Users

- Compliance managers mapping frameworks and obligations.
- Control owners maintaining control evidence.
- Auditors and assessors testing controls.
- Risk committees approving reports and attestations.
- Platform engineers composing APG compliance applications.

## Domain Model

The runtime owns these records:

- `ComplianceFramework`
- `ComplianceControl`
- `EvidenceRecord`
- `ControlAssessment`
- `ComplianceFinding`
- `ComplianceReport`
- `AttestationRecord`
- `ComplianceAuditEvent`

All internal storage keys include tenant context so repeated business IDs can
exist safely across tenants.

## Lifecycle

### Framework

1. Register framework with tenant, owner, obligations, and policy version.
2. Block missing owner, missing obligations, missing policy version, or duplicate
   tenant framework key.
3. Use the framework as the anchor for controls and reports.

### Control

1. Create control under a tenant-local framework.
2. Require name, owner, testing frequency, and DLP linkage for regulated data.
3. Use controls for evidence, assessments, findings, and coverage.

### Evidence

1. Record evidence against a tenant-local control.
2. Require source, collector, encryption, and immutable reference.
3. Use evidence age to drive assessment freshness guardrails.

### Assessment

1. Assess controls with tenant-local evidence.
2. Require a tester.
3. Route owner-tested controls to independent review.
4. Deny stale evidence unless refreshed.
5. Route failed assessments without finding linkage to review.

### Finding

1. Open findings against tenant-local controls.
2. Require owner and description.
3. Route high or critical findings without remediation plans to review.
4. Escalate overdue findings.
5. Resolve findings only when resolution evidence is attached.

### Report

1. Prepare report for tenant-local framework and period.
2. Require independent approver.
3. Require attestation statement.
4. Publish only when approval and attestation exist.
5. Block publication while critical findings remain open.

## Deterministic Rules

The contract currently exposes at least 30 rules covering:

- tenant context;
- framework ownership, obligations, policy versions, and duplicates;
- control framework, name, owner, cadence, and DLP linkage;
- evidence control, source, collector, encryption, immutable reference, and
  freshness;
- assessment tester, independence, and failed-assessment finding linkage;
- finding owner, description, remediation plan, escalation, and resolution
  evidence;
- report framework, period, preparer, independent approval, attestation, and
  publication;
- critical finding publication blocks;
- tenant isolation;
- compliance audit requirements;
- Bytewax for batch compliance mutation.

Rule decisions are one of:

- `allow`
- `require_review`
- `deny`

`deny` takes precedence over `require_review`.

## Configuration

Required configuration sections:

- `tenant_id`
- `frameworks`
- `controls`
- `evidence`
- `assessments`
- `findings`
- `reporting`
- `exceptions`
- `security`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

Key defaults:

- framework owners required;
- obligation mappings required;
- control owners required;
- testing cadence default `90` days;
- evidence freshness default `30` days;
- encrypted evidence required;
- immutable evidence reference required;
- finding remediation SLA `30` days;
- independent report approval required;
- report attestation required;
- Bytewax event stream for batch mutations;
- tenant isolation required;
- compliance audit events required.

## UI

Routes:

- `/comp/dashboard`
- `/comp/frameworks`
- `/comp/controls`
- `/comp/evidence`
- `/comp/assessments`
- `/comp/findings`
- `/comp/exceptions`
- `/comp/reports`
- `/comp/attestations`
- `/comp/exports`
- `/comp/audit`
- `/comp/settings`

View models must remain dependency-light data payloads. Browser rendering
belongs to generated applications.

## Theme

Theme name: `comp_compliance_command_center`.

Theme components:

- `framework_matrix`
- `control_card`
- `evidence_vault`
- `assessment_workbench`
- `finding_board`
- `exception_register`
- `report_builder`
- `attestation_center`
- `regulatory_export`
- `audit_timeline`

## Adapter Boundaries

Adapter keys are declared in the capability contract:

- `audit_sink`: `audl`
- `data_loss_prevention`: `dlpd`
- `encryption`: `encr`
- `authentication`: `auth`
- `security_framework`: `secu`
- `multi_tenancy`: `mten`
- `identity_federation`: `idfd`
- `zero_trust_access`: `ztna`
- `document_management`: `docm`
- `workflow`: `wflo`
- `notification`: `ntfy`
- `event_stream`: `bytewax`

Adapters must not be required for local package self-tests.

## Acceptance Criteria

- Contract exposes configuration, schema, deterministic rules, UI routes,
  theme, and adapters.
- Rule count is at least 30.
- UI route count is at least 12.
- Bytewax is the event-stream adapter.
- Service executes framework, control, evidence, assessment, finding, report,
  attestation, publication, and audit lifecycles.
- Tenant-local business IDs do not collide across tenants.
- Regulated controls require DLP linkage.
- Evidence requires encryption and immutable references.
- Stale evidence blocks assessment.
- Independent report approval is enforced.
- Critical findings block report publication.
- API helpers expose the lifecycle fields used by the service.
- View models expose all route families.
- `app.self_test()` passes.
- Focused package tests pass.
- Implementation audit and publish-plan pass.
