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
12. Audit every lifecycle decision.

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
- classification engines
- lineage graph traversal
- search indexes
- Bytewax event streams
- APG audit, auth, MDM, ETL, connector, monitoring, and notification services

Adapters must preserve the guardrail decisions produced by
`capability_contract.py`.
