# DVRL Capability Specification

## Purpose

DVRL provides APG with a first-class data virtualization capability. It lets
generated APG applications register governed virtual sources, discover schemas,
publish virtual tables, execute federated read queries, manage result caching,
record lineage, and audit all decisions through deterministic guardrails.

## Scope

DVRL owns:

- Virtual source registration and activation.
- Schema discovery and refresh review.
- Virtual table publication.
- Federated read-query request lifecycle.
- Query cache decisions.
- Virtualization policy changes.
- Source retirement and impact review.
- First-class virtualization-agent registration for AI and automation tools.
- Bytewax lifecycle-batch validation for generated-application state changes.
- Generated application UI routes, view models, and theme tokens.

DVRL integrates with:

- `etlp` for pipeline and transformation context.
- `meta` for catalog, classification, and lineage.
- `mdm` for governed data domains.
- `auth` for RBAC and actor context.
- `cach` for query-result and plan cache adapters.
- `keym` for vaulted source credentials.
- `audl` for audit events.
- `conn` and Singer adapters for physical connectivity.
- Bytewax for lifecycle stream processing.
- External AI/automation runtimes such as Codex, Claude Code, OpenCode, and Pi
  through provider-neutral agent records.

## Functional Requirements

### Source Lifecycle

DVRL must support source registration, approval, activation, schema discovery,
health status, and retirement. Source records require tenant context, owner,
supported source type, vaulted credentials, encrypted connection, and approval
before activation.

### Schema and Virtual Table Lifecycle

DVRL must support schema discovery, schema refresh review, virtual table
publication, ownership, data classification, source references, and metadata
lineage evidence.

### Query Lifecycle

DVRL must support query request creation with SQL text, actor, classification,
requested row count, cost estimate, cache decision, lineage capture, RBAC
authorization, parameterization evidence, and read-only enforcement.

### Cache Lifecycle

DVRL must support cache-result decisions with TTL enforcement, sensitive-result
blocking, cache metadata, and cache invalidation hooks for runtime adapters.

### Governance

DVRL must evaluate deterministic guardrails before side effects. Decisions must
return `allow`, `deny`, or `require_review`, matched rule names, required
actions, and evaluated context.

### Virtualization Agents

DVRL must treat AI and automation agents as first-class composition
participants. A virtualization agent record requires tenant context, supported
runtime, supported DVRL role, bounded scope, accountable owner, declared
purpose, and machine-contribution disclosure. Privileged roles such as source
review, virtual table review, query policy review, cache policy review, and
publish-gate review require human approval before the agent can participate in
governed lifecycle decisions.

DVRL does not embed vendor-specific agent clients in the lifecycle packet.
Codex, Claude Code, OpenCode, Pi, and future runtimes are represented as
declared runtime adapters that must honor DVRL guardrails, audit requirements,
and scope boundaries.

### Bytewax Lifecycle Batches

DVRL must validate lifecycle batches before generated applications apply
batched source, schema, virtual table, query, cache, policy, or agent changes.
The accepted lifecycle processor is Bytewax. Broker-specific queue-first or broker-first
processing is outside the DVRL packet boundary unless routed through a Bytewax
adapter that preserves event-time ordering, audit evidence, and deterministic
rule decisions.

## Adapter Boundaries

The dependency-light lifecycle service must not connect to physical sources or
execute SQL. Runtime connectors, query planners, execution engines, metadata
catalogs, cache stores, credential vaults, audit sinks, and Bytewax event
streams remain adapters that must honor DVRL decisions. External AI-agent
runtimes are also adapters: they can propose, review, or enrich DVRL lifecycle
decisions only through declared agent records and APG approval/audit controls.

## Acceptance Criteria

- Root `README.md`, `SPECIFICATION.md`, and `PLAN.md` describe current DVRL
  behavior and adapter boundaries.
- Contract exposes configuration, rule engine, UI routes, theme, agents,
  streaming, and adapter evidence for source, schema, query, cache, policy,
  agent, lifecycle-batch, and audit workflows.
- Generated apps can use a dependency-light lifecycle service for source,
  schema, virtual table, query, cache, policy, retirement, virtualization-agent,
  Bytewax lifecycle-batch, and audit records.
- Tests cover positive and negative guardrail paths.
- `app.py`, `semantic_model.json`, `package_manifest.json`, and
  `release_report.json` match the current contract.
- Focused compile, tests, implementation audit, publish-plan, stale marker
  scan, and diff checks pass.
