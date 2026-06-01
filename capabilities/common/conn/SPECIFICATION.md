# CONN Capability Specification

## Purpose

CONN is APG's connector and connection management capability. It gives
generated APG applications a governed control plane for connector packages,
local Singer taps, credential-safe connections, connection tests, activation,
flow composition, sync runs, schedules, replays, data quality, lineage, and
retirement.

CONN must be useful without executing a real Singer tap, opening network
connections, reading secrets, writing lineage stores, starting Bytewax workers,
calling external SaaS APIs, or executing external AI-agent runtimes. Those
systems remain adapter boundaries. The dependency-light lifecycle service must
still make connector decisions executable and auditable.

## Scope

CONN owns:

- tenant-scoped connector registration;
- connection registration, test evidence, and activation governance;
- flow composition with source/target, mapping, lineage, quality, and PII
  policy evidence;
- sync run, schedule, replay, and schema-review lifecycle records;
- marketplace, activation, schema, owner-transfer, and retirement review
  records;
- first-class connector-agent registrations for AI and automation tools that
  participate in connector review, connection review, flow design, sync
  operation, quality review, lineage review, marketplace review, credential
  review, or local Singer tap stewardship;
- Bytewax lifecycle batch validation for connector mutation streams;
- connector audit events;
- durable review evidence for policy decisions, matched rules, review reasons,
  required actions, pending review queues, and denial records;
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
- external AI and automation runtimes such as Codex, Claude Code, OpenCode,
  Pi, and future tools through governed adapter boundaries.

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

### Connector-Agent Lifecycle

CONN must model AI and automation agents as first-class connector
participants. Agent registration requires a supported runtime, supported role,
bounded scope, accountable owner, documented purpose, and
machine-contribution disclosure. Privileged roles such as connector reviewer,
connection reviewer, sync operator, quality reviewer, lineage reviewer,
marketplace reviewer, and credential reviewer require human approval evidence
before they can mutate connector state.

Supported runtimes are adapter identifiers, not embedded SDK commitments:
`codex`, `claude_code`, `opencode`, and `pi`. Future runtimes can be added by
extending the contract and adapter policy while preserving the same guardrail
shape.

### Bytewax Lifecycle Batches

CONN must validate lifecycle mutation batches before adapter side effects.
Accepted lifecycle batches must use Bytewax as the required processor. Non-
Bytewax batches are recorded as denied evidence and blocked.

### Durable Review Evidence

CONN must persist policy decisions on all generated-app connector lifecycle
records. Connectors, connections, flows, sync runs, schedules, reviews,
connector agents, lifecycle batches, and audit events carry `policy_decision`,
`matched_rules`, `review_reasons`, and `review_evidence` fields. Generated
applications must be able to compose a single pending-review queue from those
records, and denied non-Bytewax lifecycle batches must remain visible as
auditable evidence after the blocking exception.

### UI and Theme

CONN must expose generated UI models for dashboard, connectors, connections,
visual designer, sync monitor, quality, lineage, marketplace, security, audit,
rules, connector-agent roster, lifecycle-batch monitor, and settings. Theme
metadata must include compact connector, connection, flow, sync, quality,
lineage, review, security, audit, connector-agent roster, and Bytewax lifecycle
panel components.

## Guardrails

CONN decisions must return `allow`, `deny`, or `require_review`, with matched
rules and required actions. Guardrails must cover tenant context, connector
owner, runtime, source, checksum, marketplace review, connection owner,
registered connector, credential vault, encryption, secret rotation, activation
tests, production review, cross-tenant connection denial, active source/target,
mapping, lineage, quality gate, batch monitoring, batch maximum, schema review,
PII policy, webhook auth, schedule timezone, replay idempotency, destructive
delete review, retirement impact review, owner-transfer review, connector-agent
runtime, connector-agent role, agent scope, agent owner, agent purpose,
contribution disclosure, human approval for privileged agent roles, and Bytewax
lifecycle processing.

## Adapter Boundaries

The dependency-light control plane must not execute live connector operations.
Singer taps, target execution, SaaS APIs, database drivers, secret stores,
lineage stores, quality engines, monitoring sinks, audit sinks, API gateways,
service registries, external AI runtimes, and Bytewax workers are adapters
that must honor CONN decisions before side effects.

## Acceptance Criteria

- Root `README.md`, `SPECIFICATION.md`, and `PLAN.md` describe current CONN
  behavior and adapter boundaries.
- Contract exposes configuration, rules, adapters, UI, theme, and package
  evidence for connector, connection, flow, sync, schedule, replay, review,
  connector-agent, lifecycle-batch, retirement, and audit workflows.
- Contract exposes `review_evidence` metadata, and API/view-model surfaces
  expose pending review queues for generated applications.
- Generated apps can use a dependency-light service for connector lifecycle
  workflows without optional production dependencies.
- Focused tests cover positive and negative guardrail paths.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` derive from the current contract.
- Focused compile, tests, implementation audit, publish-plan, stale marker
  scan, and diff checks pass.
