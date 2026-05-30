# CONN Capability Specification

## Purpose

CONN is APG's connector and connection management capability. It gives
generated APG applications a governed control plane for connector packages,
local Singer taps, credential-safe connections, connection tests, activation,
flow composition, sync runs, schedules, replays, data quality, lineage, and
retirement.

CONN must be useful without executing a real Singer tap, opening network
connections, reading secrets, writing lineage stores, starting Bytewax workers,
or calling external SaaS APIs. Those systems remain adapter boundaries. The
dependency-light lifecycle service must still make connector decisions
executable and auditable.

## Scope

CONN owns:

- tenant-scoped connector registration;
- connection registration, test evidence, and activation governance;
- flow composition with source/target, mapping, lineage, quality, and PII
  policy evidence;
- sync run, schedule, replay, and schema-review lifecycle records;
- marketplace, activation, schema, owner-transfer, and retirement review
  records;
- connector audit events;
- generated application API helpers, UI view models, theme tokens, and package
  evidence.

CONN integrates with:

- `auth` for connector permissions;
- `keym` for credential references;
- `encr` for encryption policy;
- `audl` for audit trails;
- `moni` for health, metrics, alerts, and sync monitoring;
- `meta` for lineage metadata;
- data-quality adapters for profiling and quality gates;
- `regy` for connector and service registry publication;
- `apig` for API exposure when needed;
- Bytewax-backed streams for connector lifecycle events.

## Functional Requirements

### Connector Lifecycle

CONN must register connectors with tenant, owner, runtime, source reference,
checksum, verification state, and metadata. Unverified connector packages
require marketplace review before connection use. Webhook connectors require an
auth policy.

### Connection Lifecycle

CONN must register connections with connector reference, owner, environment,
credential vault reference, encryption evidence, and metadata. Activation
requires a passed connection test and secret rotation evidence. Production
activation requires review evidence.

### Flow Lifecycle

CONN must create flows only when source and target connections are active.
Flows require mapping evidence, lineage capture, quality gate evidence, and PII
policy evidence when sensitive data is detected.

### Sync Lifecycle

CONN must start sync runs with mode, batch size, monitoring evidence, and schema
change review when needed. Oversized batches are denied. Large batches require
monitoring. Completed sync runs keep quality score and processed-record
evidence.

### Schedule and Replay Lifecycle

CONN must schedule flows with timezone evidence. Replay operations require
idempotency keys.

### Retirement Lifecycle

CONN must retire connections only after impact review evidence. Retirement
keeps historical records and emits audit evidence.

### UI and Theme

CONN must expose generated UI models for dashboard, connectors, connections,
visual designer, sync monitor, quality, lineage, marketplace, security, audit,
rules, and settings. Theme metadata must include compact connector, connection,
flow, sync, quality, lineage, review, security, and audit components.

## Guardrails

CONN decisions must return `allow`, `deny`, or `require_review`, with matched
rules and required actions. Guardrails must cover tenant context, connector
owner, runtime, source, checksum, marketplace review, connection owner,
registered connector, credential vault, encryption, secret rotation, activation
tests, production review, cross-tenant connection denial, active source/target,
mapping, lineage, quality gate, batch monitoring, batch maximum, schema review,
PII policy, webhook auth, schedule timezone, replay idempotency, destructive
delete review, retirement impact review, and owner-transfer review.

## Adapter Boundaries

The dependency-light control plane must not execute live connector operations.
Singer taps, target execution, SaaS APIs, database drivers, secret stores,
lineage stores, quality engines, monitoring sinks, audit sinks, API gateways,
service registries, and Bytewax workers are adapters that must honor CONN
decisions before side effects.

## Acceptance Criteria

- Root `README.md`, `SPECIFICATION.md`, and `PLAN.md` describe current CONN
  behavior and adapter boundaries.
- Contract exposes configuration, rules, adapters, UI, theme, and package
  evidence for connector, connection, flow, sync, schedule, replay, review,
  retirement, and audit workflows.
- Generated apps can use a dependency-light service for connector lifecycle
  workflows without optional production dependencies.
- Focused tests cover positive and negative guardrail paths.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` derive from the current contract.
- Focused compile, tests, implementation audit, publish-plan, stale marker
  scan, and diff checks pass.
