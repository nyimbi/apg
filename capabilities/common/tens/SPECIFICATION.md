# TENS Capability Specification

## 1. Identity

- Capability ID: `tens`
- Name: Tenants Legacy
- Category: common platform capability
- Runtime target: APG Python capability package
- Primary users: tenant migration teams, platform operators, access governance teams, generated application administrators

TENS gives generated applications a governed compatibility layer for legacy tenants. It tracks legacy tenant records, validates mappings to APG tenants, enforces access-boundary evidence, governs migration plans, records deprecation plans, composes AI-assisted review lanes, and publishes lifecycle events through Bytewax metadata.

## 2. Scope

TENS owns the executable lifecycle for:

- legacy tenant registration;
- source-system and compatibility-scope lineage;
- mapping to APG tenant IDs;
- access boundary, role mapping, isolation, and privileged review evidence;
- migration approval, rollback, completion, and post-migration validation;
- deprecation planning;
- governed AI agent composition for mapping, boundary, migration, deprecation, compatibility, and audit review;
- audit and policy surfaces for generated applications.

Identity providers, legacy directories, tenant catalogs, role stores, migration engines, audit sinks, and approval systems stay behind adapters.

## 3. Provided Services

- `legacy_tenant_registry`
- `tenant_mapping`
- `migration_controls`
- `access_boundaries`
- `deprecation_governance`
- `tens_agents`

## 4. Required Services

- `mten` for APG multi-tenant boundaries
- `auth` for authorization and access-boundary checks
- `audl` for durable audit publication
- `idfd` for identity federation and legacy identity mapping
- `usrm` for user and role mapping context

Optional integrations include consent, legacy identity directories, tenant catalog imports, migration engines, approval systems, and compliance adapters.

## 5. Domain Model

### Legacy Tenant

A legacy tenant records tenant context, legacy tenant ID, source system, owner, compatibility scope, activity age, status, required actions, and timestamps.

States:

- `active`
- `stale`
- `mapped`
- `migration_ready`
- `migrated`
- `deprecated`
- `blocked`

### Tenant Mapping

A tenant mapping links a legacy tenant record to an APG tenant ID with validation owner and validation evidence.

### Access Boundary

An access boundary records auth-boundary evidence, role mapping evidence, isolation validation, privileged access review, actor, and status.

### Migration Plan

A migration plan links a legacy tenant and mapping to owner, approval, rollback, post-migration validation, status, and completion timestamp.

### Deprecation Plan

A deprecation plan records owner, deprecation evidence, target date, and status for retiring legacy tenant compatibility.

### TENS Agent

A TENS agent is a first-class composition element with tenant, name, runtime, role, scope, owner, status, and human approval policy.

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `tenant_mapper`
- `boundary_reviewer`
- `migration_reviewer`
- `deprecation_reviewer`
- `compatibility_reviewer`
- `audit_reviewer`

Agents can review and prepare tenant mapping work, but privileged tenant actions require human approval.

### Audit Event

An audit event records tenant, event type, subject, message, actor, severity, metadata, and timestamp.

## 6. Rule Engine

The deterministic rule engine enforces:

- tenant context on every operation;
- owner, source system, and compatibility scope on legacy tenant registration;
- mapping validation and Bytewax lifecycle routing;
- migration approval, rollback, post-validation, and Bytewax lifecycle routing;
- access boundary, role mapping, tenant isolation, and privileged review evidence;
- stale tenant review after the configured activity age;
- supported agent runtime and role;
- human approval for privileged agent-driven tenant actions;
- Bytewax coordination for batch tenant mapping.

Rules return `allow`, `require_review`, or `deny` with required actions.

## 7. Workflows

### Registration

1. Register legacy tenant with source system, owner, compatibility scope, and activity age.
2. Require review for stale tenants.
3. Emit `legacy_tenant_registered`.

### Mapping

1. Map legacy tenant to APG tenant ID.
2. Require validation evidence.
3. Require Bytewax event stream.
4. Emit `tenant_mapped`.

### Boundary Validation

1. Attach auth boundary reference.
2. Attach legacy role mapping evidence.
3. Attach tenant isolation validation.
4. Attach privileged access review.
5. Emit `boundary_validated`.

### Migration

1. Create migration plan with mapping, owner, approval, rollback, and post-migration validation references.
2. Complete migration with post-migration validation and Bytewax event stream.
3. Emit `migration_plan_created` and `migration_completed`.

### Deprecation

1. Attach deprecation plan evidence and target date.
2. Mark legacy tenant deprecated.
3. Emit `deprecation_planned`.

### Agent Workflow

1. Register agent with supported runtime and role.
2. Validate privileged agent-driven tenant actions.
3. Deny privileged action without human approval.
4. Emit `tens_agent_registered`.

## 8. UI Contract

TENS exposes APG Python view models for:

- `/tens/dashboard`
- `/tens/tenants`
- `/tens/mappings`
- `/tens/migrations`
- `/tens/boundaries`
- `/tens/deprecation`
- `/tens/agents`
- `/tens/policy`
- `/tens/audit`
- `/tens/settings`

Generated UIs should prioritize lineage, mapping status, boundary evidence, migration readiness, deprecation plans, and unresolved guardrails.

## 9. Theming

The default theme is `tens_legacy_tenant_migration`. It defines compact density, legacy status pills, migration bands, validation chips, approval chips, isolation chips, review lanes, and guardrail chips.

## 10. Event Stream

TENS lifecycle events use Bytewax:

- processor: `bytewax`
- stream: `apg.tens.lifecycle`
- key: `tenant_id`

Events:

- `legacy_tenant_registered`
- `tenant_mapped`
- `boundary_validated`
- `migration_plan_created`
- `migration_completed`
- `deprecation_planned`
- `tens_agent_registered`

## 11. Acceptance Criteria

- Contract exposes configuration, schema, rules, UI, theme, services, dependencies, and streaming metadata.
- Service executes registration, mapping, boundary, migration, deprecation, agent, batch-validation, and audit lifecycles without external dependencies.
- Rules deny invalid tenant, owner, source, compatibility scope, validation, stream, approval, rollback, boundary, role, isolation, privileged review, agent, and human-approval states.
- API helpers expose the executable lifecycle.
- View models expose dashboard, registry, mapping, migration, boundary, deprecation, agents, policy, audit, and settings.
- Generated package artifacts reflect the current contract.
- Focused package verification passes.
