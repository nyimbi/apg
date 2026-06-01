# FinTech Compliance Automation Specification

## Purpose

FinTech Compliance Automation gives APG applications a first-class compliance
operating surface. It turns regulatory obligations, controls, testing,
evidence, attestations, issues, remediation, reports, reviews, and AI-agent
support into composable application capabilities.

## Functional Scope

- Register obligations for supported regulatory frameworks and obligation
  types.
- Map controls to obligations with ownership, evidence, and frequency.
- Record compliance checks against obligations and controls.
- Attach evidence with source and retention metadata.
- Record attestations and review statuses.
- Open compliance issues and record remediation plans.
- Publish compliance reports with framework, period, evidence, and approver.
- Register provider-neutral AI agents with supported runtimes and roles.
- Publish UI routes, theme metadata, and Bytewax lifecycle metadata.

## Guardrails

- Every write requires tenant context and policy evidence.
- Obligations require supported framework, supported type, owner, evidence, and
  effective date.
- Controls require an obligation, supported type, owner, evidence, and
  frequency.
- Checks require obligation, control, supported check type, subject, result, and
  evidence when the check fails.
- Evidence requires supported type, reference, source, and retention period.
- Attestations require obligation, attestor, supported status, and evidence.
- Issues require obligation, supported severity, owner, evidence, and due date.
- High-impact remediation requires approval evidence.
- Reports require supported type, supported framework, period, evidence, and
  approver.
- Reviews require supported status, reviewer, and evidence.
- Batch lifecycle events require Bytewax routing.
- Privileged AI-agent actions require human approval.

## Non-Goals

- No live regulator filing, document signing, external GRC suite integration,
  payment capture, ledger posting, or durable worker topology is embedded in
  this package.
- External systems remain behind APG adapter contracts.
