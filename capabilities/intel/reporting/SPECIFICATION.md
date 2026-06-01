# Intelligence Reporting Specification

## Purpose

Intelligence Reporting lets APG applications produce, review, approve, and
disseminate intelligence products with a complete chain of authority,
classification, citations, human approvals, distribution evidence, and
lifecycle events. It is designed for analysts, reviewers, approvers, and
operations teams that need high-trust reports rather than unmanaged documents.

## Users

- Analysts who draft sections and assemble products.
- Source reviewers who check citations and supporting evidence.
- Classification and legal reviewers who approve release.
- Operations teams that receive approved briefings and notices.
- AI-agent supervisors who delegate bounded writing, citation review,
  classification review, editorial review, distribution review, and briefing
  preparation work.

## Functional Scope

- Authorities: lawful reporting mandates with scope, classification, approver,
  expiry, and evidence.
- Workspaces: governed reporting containers for strategic, tactical, threat,
  incident, investigative, executive, and partner products.
- Templates: reusable structures for briefs, advisories, bulletins, estimates,
  situation reports, watchlists, case summaries, and executive summaries.
- Products: concrete intelligence reports with title, author, classification,
  template, and evidence.
- Sections: report content units with section type, confidence, and evidence.
- Citations: source, case, graph, RAG, geospatial, model-output, and analyst
  note references.
- Approvals: editorial, classification, legal, operational, partner-release,
  and executive-release approval records.
- Distributions: approved dissemination to internal, partner, executive,
  field, watch-center, or case-team recipients.
- Publications: approved publication through portal, email digest,
  notification, case file, secure export, or briefing pack channels.
- Reviews: human review outcomes for lifecycle artifacts.
- AI agents: provider-neutral runtimes with bounded roles and explicit scope.

## Out Of Scope

This package does not render PDF/HTML, send live notifications, write case
files, mutate graph/RAG stores, perform live source extraction, publish to
external portals, or run durable streaming topologies. Those remain adapter
responsibilities until their contracts are explicit.

## Lifecycle

1. Record authority.
2. Create reporting workspace.
3. Record template.
4. Record product.
5. Record sections.
6. Record citations for claims.
7. Record approval.
8. Record approved distribution.
9. Record approved publication.
10. Record human review.
11. Register bounded AI agents.
12. Route lifecycle batches through Bytewax.

## Rule Engine

The deterministic rule engine denies missing tenant context, unsupported
taxonomy values, missing evidence, missing authority, invalid confidence,
missing authors, missing citations, missing approvals, unsupported statuses,
non-Bytewax batches, unsupported agent runtimes or roles, missing agent scope,
privileged agent actions without approval, uncited claims, classification
downgrades, source fabrication, privacy bypasses, autonomous publication, and
unapproved distribution.

## UI And Theme

The capability exposes APG Python UI route metadata for dashboard,
authorities, workspaces, templates, products, sections, citations, approvals,
distributions, publications, reviews, agents, and settings. The theme uses
compact, work-focused tokens under `intel_reporting_control`.

## Adapter Boundaries

Generated applications compose this capability with auth, audit, notification,
NLP, graph, RAG, and geospatial capabilities. Production integrations should
bind persistent document storage, rendering, exports, notifications, graph/RAG
writes, case files, and durable Bytewax workers through adapters without
bypassing this package's deterministic rules.

