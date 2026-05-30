# ENVM Environment Management Specification

## Purpose

The ENVM capability (`envm`) lets generated APG applications compose
tenant-scoped environments, promotion paths, promotion runs, configuration
drift, secret scopes, environment policy, audit evidence, visual route
metadata, theme metadata, Bytewax stream governance, and AI-agent assistance
into ERP, SaaS, infrastructure, compliance, and release-management
applications.

This package owns the executable contract, deterministic guardrails,
dependency-light service, API helpers, view models, UI route metadata, theme
metadata, Bytewax stream declaration, generated semantic evidence, and focused
proof commands. Deployment providers, live configuration stores, secret vaults,
runtime access checks, monitoring pipelines, and stream-worker deployments
remain adapter concerns.

## Users And Jobs

- Platform owners register environments with stage, region, configuration
  source, RBAC policy, secret-scope policy, and ownership.
- Release managers define governed promotion paths and execute promotions.
- Operations teams monitor drift between declared and observed state.
- Security reviewers inspect secret scopes, access roles, production approval,
  and audit evidence.
- Platform engineers bind deployment, identity, configuration, secret,
  monitoring, audit, and Bytewax workers.
- AI agents assist with environment review, promotion review, drift review,
  secret-scope review, and policy review under explicit registration and
  disclosure.

## Capability Boundary

`envm` provides:

- environment inventory;
- environment promotion paths and runs;
- configuration drift reporting;
- secret-scope governance;
- environment policy metadata;
- AI ENVM-agent registration and policy enforcement;
- Bytewax stream metadata for batch environment mutation.

`envm` requires:

- `auth` for identity and permission context;
- `conf` for tenant configuration;
- `audl` for durable audit evidence;
- `depl` for deployment links and release artifacts;
- `keym` for secret references;
- `moni` for operational monitoring.

## Lifecycle

Environment lifecycle:

1. An environment is registered with tenant, name, owner, stage, region,
   configuration source, RBAC policy, and secret-scope policy.
2. Unsupported stages are rejected.
3. Production environments require approval evidence and are locked by default.
4. Environment changes record audit evidence.

Promotion lifecycle:

1. A promotion path links source, target, deployment link, rollback
   environment, and approval state.
2. Production targets require approval.
3. Promotion runs require an artifact reference and a governed path.
4. Promotion events record audit evidence.

Drift lifecycle:

1. A drift report compares declared and observed environment versions.
2. Drift percentage is calculated from changed and total items.
3. Drift above the threshold requires review.
4. Remediation remains an adapter concern until linked to deployment and
   configuration providers.

Secret-scope lifecycle:

1. A secret scope references one managed environment.
2. The scope requires policy reference, secret references, and access roles.
3. Secret values remain in `keym`; ENVM stores references and governance
   metadata only.

AI-agent lifecycle:

1. Agent is registered with runtime, role, scope, tenant, and disclosure.
2. Runtime must be one of `codex`, `claude_code`, `opencode`, or `pi`.
3. Role must be one of the configured ENVM roles.
4. Agent contributions are audit-visible and cannot bypass policy decisions.

## Rule Engine

Rules must deny or require review for:

- missing tenant context;
- missing environment owner, region policy, configuration source, or RBAC
  policy;
- production change without approval;
- promotion without path or artifact reference;
- secret scope without policy, secret references, or access roles;
- high drift without review;
- unregistered, unsupported, unscoped, or undisclosed AI agents;
- lifecycle state changes without audit evidence;
- batch environment mutations that do not use Bytewax.

## UI And Theme

The APG Python UI contract exposes dashboard, environments, promotion, drift,
secrets, agents, policies, rules, analytics, audit, and settings routes. The
theme uses compact operational density with distinct treatments for environment
grids, promotion flows, drift dashboards, secret scopes, ENVM-agent scope,
stream health, and audit events.

## Streaming

Batch environment mutation must use Bytewax. The stream topic is
`apg.envm.lifecycle`, and state covers environments, promotion paths, promotion
runs, drift reports, secret scopes, ENVM agents, and audit events. Live Bytewax
topology deployment is an adapter concern, but the package declares and
enforces the guardrail.

## Adapter Boundaries

Adapters must handle:

- deployment provider integration through `depl`;
- durable configuration stores through `conf`;
- secret value storage through `keym`;
- authentication and permission checks through `auth`;
- audit durability through `audl`;
- monitoring and alerting through `moni`;
- Bytewax lifecycle topology and operational monitoring.

## Acceptance Gates

- Contract validates through the APG capability registry.
- Configuration schema includes environments, promotion, drift, secrets, ENVM
  agents, governance, observability, adapters, UI, and theme.
- Rules cover environment, promotion, drift, secret, agent, audit, and Bytewax
  guardrails.
- Service can register environments, create promotion paths, run promotions,
  record drift, register secret scopes, register agents, summarize state, and
  validate batch mutation streams.
- API helpers and view models expose the same lifecycle surfaces.
- Generated semantic evidence exposes provides/requires, routes, rules, theme,
  and streaming.
- README, specification, plan, progress log, focused tests, implementation
  audit, publish plan, and stale-marker scan are current.
