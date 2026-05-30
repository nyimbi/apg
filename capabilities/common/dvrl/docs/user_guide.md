# DVRL User Guide

DVRL lets APG users compose governed data access into generated applications.
The current packet focuses on lifecycle control: source registration, source
activation, schema review, virtual table publication, query guardrails, cache
decisions, policy review, source retirement, and audit evidence.

## Core Workflow

1. Register a virtual source with owner, supported type, vaulted credentials,
   and encrypted connection evidence.
2. Activate the source after approval evidence is recorded.
3. Refresh or review source schema metadata.
4. Publish virtual tables with owner and classification evidence.
5. Submit federated read-query requests with parameterization, RBAC,
   classification, lineage, cost, join, row-limit, and cache evidence.
6. Cache allowed query results within the configured TTL limit.
7. Review policy changes before activation.
8. Retire sources only after impact review.

## Generated UI Surfaces

- Dashboard
- Query workbench
- Source manager
- Schema browser
- Virtual table catalog
- Federation map
- Policy console
- Cache console
- Metrics
- Adapter health
- Audit timeline
- Settings

## Guardrail Outcomes

Each lifecycle operation returns one of:

- `allow`: the operation can proceed.
- `deny`: the operation is blocked.
- `require_review`: the operation is recorded as pending review.

Responses include matched rule names and the evaluated context so generated
applications can show clear status and remediation actions.

## Runtime Boundary

DVRL lifecycle operations do not open live source connections. Production
deployments bind physical connectors, query planners, execution engines,
metadata catalogs, cache stores, credential vaults, audit sinks, and Bytewax
event streams through adapters that must honor DVRL decisions.
