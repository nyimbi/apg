# APG Master Data Management Capability Summary

**Capability ID**: `common/mdm`  
**Version**: 1.0.0  
**Status**: Executable capability packet
**Primary runtime**: `service.MdmService` for generated applications,
`service.MDMService` for database-backed production adapters

## Summary

The APG Master Data Management capability provides tenant-scoped entity
governance for applications that need dependable customers, products,
suppliers, employees, locations, assets, accounts, contracts, organizations, and
custom master-data records.

The current packet focuses on executable composition: contract data, lifecycle
service methods, guardrails, UI route metadata, theme tokens, view models,
semantic model generation, package tests, and adapter boundaries.

## Lifecycle

1. Register entity.
2. Assess quality.
3. Detect or record duplicate candidates.
4. Review duplicate candidates through stewardship.
5. Create golden records with survivorship policies.
6. Evaluate merge requests.
7. Map source-system cross references.
8. Publish mastered records after readiness checks.
9. Retire records with lineage evidence.
10. Audit every lifecycle decision.

## Guardrails

The capability denies or routes decisions when:

- Tenant context is missing.
- Entity type is not configured.
- Business key is missing.
- Restricted data lacks owner, audit evidence, or classification evidence.
- Quality scores are outside the configured range.
- Publish is attempted without owner or current quality evidence.
- Quality is below the blocking threshold.
- Duplicate candidates require steward review.
- Golden-record merge lacks survivorship policy.
- Conflicted merge lacks an independent steward.
- Cross-reference changes lack source-system evidence.
- Retirement lacks lineage evidence.
- Review decisions lack notes.

## UI And Theme

Generated applications can compose these surfaces:

- Dashboard
- Entity workbench
- Golden-record manager
- Quality console
- Duplicate review queue
- Stewardship queue
- Lineage trace
- Cross-reference console
- Publish readiness console
- Analytics
- Audit timeline
- Adapter health
- Settings

The theme contract exposes compact golden-record console tokens and component
metadata for cards, review queues, quality panels, lineage traces, stewardship
decisions, cross-reference matrices, publish readiness, audit timelines, and
adapter status panels.

## Adapter Boundary

The packet does not require a database, AI engine, cache, or event runtime to
be useful. Production adapters can connect those systems behind the same
contract:

- database persistence through `MDMService`
- matching and quality engines
- metadata catalog synchronization
- graph lineage persistence
- Bytewax event streams
- cache and audit services

Adapters must preserve the guardrail decisions produced by
`capability_contract.py`.
