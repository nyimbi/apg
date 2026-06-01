# APG Metadata Management Capability Summary

**Capability ID**: `common/meta`
**Version**: 1.0.0
**Status**: Executable capability packet
**Primary runtime**: `service.MetaService` for generated applications,
`service.APGMetadataService` for production adapters

## Summary

The APG Metadata Management capability provides tenant-scoped metadata catalog
governance for generated applications. It covers metadata asset registration,
approved discovery, classification review, lineage capture, quality assessment,
certification, glossary ownership, publication, retirement, generated UI state,
and audit evidence.

The current packet focuses on executable composition: contract data, lifecycle
service methods, guardrails, UI route metadata, theme tokens, view models,
semantic model generation, package tests, and adapter boundaries.

It also makes AI/catalog agents first-class META participants. Agents can be
registered per tenant with supported runtime, role, scope, owner, purpose,
machine-contribution disclosure, and human-approval metadata. Bytewax is the
required lifecycle processing engine for this packet.

The packet now preserves durable review evidence across the generated-app
control plane. Reviewable records expose `policy_decision`, `matched_rules`,
`review_reasons`, and `review_evidence`, and generated UI models can compose
pending-review queues without replaying rule evaluation.

The database connector surface also includes executable fixture-backed Oracle,
SQL Server, Redis, and BigQuery connectors. These connectors import without live
vendor drivers and support connection checks, asset discovery, schema lookup,
and sample data reads from configured metadata fixtures.

## Lifecycle

1. Register asset.
2. Schedule approved discovery.
3. Record discovery results.
4. Classify sensitive assets.
5. Review low-confidence classifications.
6. Capture lineage between registered assets.
7. Assess metadata quality.
8. Request certification.
9. Publish governed assets.
10. Register glossary terms.
11. Retire assets with impact evidence.
12. Register governed catalog agents for discovery, classification, lineage,
    glossary, certification, and publish-gate workflows.
13. Validate lifecycle batches through Bytewax.
14. Audit every lifecycle decision.

## Guardrails

The capability denies or routes decisions when:

- Tenant context is missing.
- Asset type is unsupported.
- Business key or source system is missing.
- Published assets lack an owner or quality evidence.
- Restricted assets lack classification or steward assignment.
- Certification lacks lineage evidence or quality threshold.
- Classification review lacks notes.
- Low-confidence classification needs steward review.
- Discovery connector is not approved.
- Discovery schedule review is stale.
- Lineage references unregistered assets.
- Lineage depth exceeds configured review threshold.
- Glossary term lacks an owner.
- Retirement lacks impact-analysis evidence.
- Stale assets need freshness review before certification.
- Catalog-agent runtime or role is unsupported.
- Catalog-agent scope, owner, purpose, or contribution disclosure is missing.
- Privileged catalog-agent roles are registered without human approval; otherwise
  valid registrations are preserved as `pending_review` evidence.
- Lifecycle batches are submitted through anything other than Bytewax.
  Non-Bytewax submissions persist `denied` evidence before raising
  `PermissionError`.

## UI And Theme

Generated applications can compose these surfaces:

- Dashboard
- Asset catalog
- Discovery console
- Lineage viewer
- Classification review
- Quality console
- Certification queue
- Business glossary
- Impact analysis
- Search
- Audit timeline
- Adapter health
- Catalog-agent roster
- Bytewax lifecycle monitor
- Settings

The theme contract exposes compact catalog-console tokens and component metadata
for asset cards, lineage graphs, classification queues, discovery timelines,
certification queues, glossary panels, impact graphs, audit timelines, and
adapter status panels.

## Adapter Boundary

The packet does not require a database, discovery connector, AI classifier,
graph store, search index, or stream runtime to be useful. Production adapters
can connect those systems behind the same contract:

- metadata store persistence
- discovery connector execution
- fixture-backed Oracle, SQL Server, Redis, and BigQuery catalog metadata for
  generated apps
- classification engines
- lineage graph traversal
- search indexes
- Bytewax lifecycle streams
- APG audit, auth, MDM, ETL, connector, monitoring, and notification services

Adapters must preserve the guardrail decisions produced by
`capability_contract.py`.

Agent runtime adapters may integrate Codex, Claude Code, opencode, Pi, or later
providers. They must remain adapter-level integrations and must not replace the
META contract as the authority for allowed actions.
