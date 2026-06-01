# Threat Intelligence Specification

## Purpose

Threat Intelligence lets APG applications collect, curate, assess, and act on
threat information with a complete chain of authority, source lineage,
evidence, analyst review, approval, and lifecycle events. It is designed for
security, intelligence, fraud, risk, and operations teams that need to turn
signals into governed threat products and mitigations.

## Users

- Analysts who register sources, indicators, actors, campaigns, and assessments.
- Reviewers who approve threat reports and mitigation actions.
- Operations teams that consume mitigations and watchlist outputs.
- AI-agent supervisors who delegate bounded source triage, indicator curation,
  actor analysis, assessment review, mitigation review, and report writing.

## Functional Scope

- Authorities: lawful threat-intelligence mandates with scope, classification,
  approver, expiry, and evidence.
- Workspaces: governed analytical containers for strategic, cyber, physical,
  fraud, geopolitical, insider, and supply-chain threat work.
- Sources: OSINT, SIGINT, HUMINT, GEOINT, CYBINT, FININT, partner reports, and
  sensor feeds with custodian and lineage references.
- Indicators: IOCs, tactics, techniques, procedures, behaviors,
  vulnerabilities, infrastructure, narratives, and financial signals.
- Actors: state actors, criminal groups, insiders, hacktivists, terrorist
  networks, competitors, and unknown actors.
- Campaigns: intrusion, fraud, disinformation, physical-threat, insider, and
  supply-chain campaigns.
- Assessments: threat profiles, risk, priority, attribution, intent, and
  capability assessments.
- Reports: briefs, advisories, bulletins, estimates, watchlists, and situation
  reports.
- Mitigations: monitor, block, patch, investigate, harden, disrupt, escalate,
  and notify actions.
- Reviews: human review records for lifecycle artifacts.
- AI agents: provider-neutral runtimes with bounded roles and explicit scope.

## Out Of Scope

This package does not execute live enrichment, graph writes, persistent case
updates, RAG indexing, notification delivery, report rendering, threat-feed
pulls, sandbox detonation, automated takedown, or durable streaming topologies.
Those remain adapter responsibilities until their contracts are explicit.

## Lifecycle

1. Record authority.
2. Create threat workspace.
3. Register source with custodian, lineage, and evidence.
4. Record indicators and actors.
5. Record campaigns tied to actors.
6. Record analyst assessment.
7. Approve and record report.
8. Approve and record mitigation.
9. Record human review.
10. Register bounded AI agents.
11. Route lifecycle batches through Bytewax.

## Rule Engine

The deterministic rule engine denies missing tenant context, unsupported
taxonomy values, missing evidence, missing authority, missing source lineage,
invalid confidence scores, unsupported risk levels, missing analysts, missing
approval, non-Bytewax batches, unsupported agent runtimes or roles, missing
agent scope, privileged agent actions without approval, unsupported
attribution, fabricated indicators, source tampering, privacy bypasses,
autonomous mitigation, and unapproved publication.

## UI And Theme

The capability exposes APG Python UI route metadata for dashboard,
authorities, workspaces, sources, indicators, actors, campaigns, assessments,
reports, mitigations, reviews, agents, and settings. The theme uses compact,
work-focused tokens under `intel_threats_control`.

## Adapter Boundaries

Generated applications compose this capability with auth, audit, notification,
NLP, graph, RAG, and geospatial capabilities. Production integrations should
bind storage, feed ingestion, enrichment, graph mutation, retrieval indexing,
notification delivery, rendered UI, case management, and durable Bytewax
workers through adapters without bypassing this package's rules.

