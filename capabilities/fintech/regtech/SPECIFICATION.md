# Regulatory Technology Specification

## Purpose

Regulatory Technology gives APG applications a first-class regulatory operating
surface. It turns source monitoring, change intake, obligation and policy mapping, impact
assessment, filing preparation, submission evidence, inquiry response, reviews,
and AI-agent support into composable application capabilities.

## Functional Scope

- Register regulatory sources by regulator, jurisdiction, reference, owner, and
  evidence.
- Record regulatory changes with source, framework, change type, effective
  date, severity, and evidence.
- Map changes to obligations and policy references.
- Assess regulatory impact against affected APG capabilities with reviewer
  evidence.
- Prepare filings with framework, period, evidence, and owner metadata.
- Record submissions with channel, submitter, timestamp, and acknowledgment.
- Open regulator inquiries and record approved responses.
- Record reviews for any regulatory artifact.
- Register provider-neutral AI agents with supported runtimes and roles.
- Publish UI routes, theme metadata, and Bytewax lifecycle metadata.

## Guardrails

- Every write requires tenant context and policy evidence.
- Sources require supported regulator, supported jurisdiction, source
  reference, owner, and evidence.
- Changes require existing source, supported framework, supported type,
  effective date, supported severity, and evidence.
- Obligation mappings require existing change, obligation reference, policy
  reference, owner, and due date.
- Impact assessments require existing change, affected capability, supported
  risk rating, reviewer, and evidence.
- Filings require supported framework, supported type, period, evidence, and
  owner.
- Submissions require filing, supported channel, submitter, submitted timestamp,
  and acknowledgment.
- Inquiries require regulator, reference, supported severity, due date, and
  evidence.
- Responses require existing inquiry, responder, response reference, and approval
  reference.
- Reviews require supported status, reviewer, and evidence.
- Batch lifecycle events require Bytewax routing.
- Privileged AI-agent actions require human approval.

## Non-Goals

- No live regulator portal submission, external regulatory-feed subscription,
  document signing, external GRC suite integration, or durable worker topology
  is embedded in this package.
- External systems remain behind APG adapter contracts.
