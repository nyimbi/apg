# Connection Management (CONN) Capability Summary

CONN provides APG applications with a governed connector and data-flow control
plane for local Singer taps, secured connections, flow composition, sync runs,
schedules, replays, lineage, quality gates, first-class connector-agent
composition, Bytewax lifecycle batches, and audit evidence.

## Runtime Shape

- `conn_runtime.py` is the dependency-light generated-app lifecycle runtime.
- `service.py` is the production-oriented connection manager and flow executor.
- `singer_runtime.py` and `singer_taps/` contain local Singer runtime surfaces.
- `api.py` exposes FastAPI routes and generated-app helper functions.
- `views.py` and frontend files provide production UI surfaces.
- `view_models.py` exposes dependency-light UI data for generated apps.
- `app.py` derives semantic model and package evidence from the contract.

## Lifecycle

1. Register a connector with tenant, owner, runtime, source reference, checksum,
   and verification state.
2. Register a connection using a registered connector and credential vault
   evidence.
3. Record a passed connection test and secret rotation evidence before
   activation.
4. Create flows with active source/target connections, mappings, lineage, and
   quality gates.
5. Start sync runs with batch, monitoring, and schema-review evidence.
6. Schedule flows with timezone evidence and replay syncs with idempotency
   evidence.
7. Register governed AI and automation agents with runtime, role, scope, owner,
   purpose, contribution disclosure, and human approval for privileged roles.
8. Validate connector lifecycle mutation batches through Bytewax before adapter
   side effects.
9. Retire connections only after impact review evidence.
10. Emit audit events for lifecycle decisions.

## Guardrails

CONN guardrails deny missing tenant context, connector owner/runtime/source/
checksum, connection owner, registered connector, credential vault, credential
encryption, secret rotation, activation test, cross-tenant connection,
inactive source or target, missing mapping, lineage, quality gate, batch
monitoring, oversized batches, PII policy, webhook auth, schedule timezone,
replay idempotency, destructive delete review, and retirement impact review.
They also deny unsupported connector-agent runtimes, unsupported agent roles,
missing agent scope, missing owner, missing purpose, missing contribution
disclosure, and non-Bytewax connector lifecycle batches.

CONN guardrails require review for unverified connector packages, production
activation, schema changes, owner transfers, and privileged connector-agent
roles that lack human approval evidence.

## Adapter Boundaries

Generated-app CONN does not run Singer taps, open network/database/SaaS
connections, read secrets, write lineage stores, execute Bytewax flows, or
perform external side effects. Production adapters for `auth`, `keym`, `encr`,
`audl`, `moni`, `meta`, data quality, `regy`, `apig`, local Singer runtime, and
external agent runtimes, and Bytewax must honor CONN decisions before side
effects.
