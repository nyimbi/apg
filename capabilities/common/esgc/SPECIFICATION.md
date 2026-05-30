# ESGC ESG and Carbon Tracking Specification

## Purpose

The ESGC capability (`esgc`) lets generated APG applications compose
tenant-scoped emissions inventories, approved factor libraries, activity
emissions, sustainability reporting, target tracking, compliance evidence,
visual route metadata, theme metadata, Bytewax stream governance, and AI-agent
assistance into ERP, manufacturing, energy, logistics, retail, and governance
applications.

This package owns the executable contract, deterministic guardrails,
dependency-light service, API helpers, view models, UI route metadata, theme
metadata, Bytewax stream declaration, generated semantic evidence, and focused
proof commands. Meter integrations, forecasting models, compliance filings,
durable audit stores, geospatial providers, and stream-worker deployments
remain adapter concerns.

## Users And Jobs

- Sustainability teams create inventories with organization owner, reporting
  year, boundary, geospatial boundary, and compliance framework.
- Data stewards register approved emission factors with source evidence,
  version, scope, unit, and conversion rate.
- Operators record activity data from meters, invoices, logistics events, or
  operational systems.
- Reviewers inspect anomalies, factor sources, report approvals, compliance
  mapping, and audit evidence.
- Executives and operations leaders track reduction targets and progress.
- Platform engineers bind geospatial, audit, forecast, compliance, metering,
  and Bytewax workers.
- AI agents assist with inventory review, factor review, activity review,
  report review, and target review under explicit registration and disclosure.

## Capability Boundary

`esgc` provides:

- emissions inventory;
- factor library;
- activity emissions;
- sustainability reporting;
- target tracking;
- ESG evidence metadata;
- AI ESGC-agent registration and policy enforcement;
- Bytewax stream metadata for batch ESG mutation.

`esgc` requires:

- `auth` for identity and permission context;
- `conf` for tenant configuration;
- `audl` for durable audit evidence;
- `geos` for reporting boundaries;
- `pred` for forecasts;
- `comp` for compliance mapping.

## Lifecycle

Inventory lifecycle:

1. An inventory is created with tenant, organization, owner, reporting year,
   boundary, geospatial boundary, and compliance framework.
2. Missing owner or boundary denies creation.
3. Inventory creation records audit evidence.

Factor lifecycle:

1. A factor is registered with tenant, name, scope, unit, conversion rate,
   source, evidence, version, and approval state.
2. Unsupported scopes are rejected.
3. Unapproved sources, missing source evidence, or missing version deny
   registration.

Activity lifecycle:

1. An activity references an inventory and approved factor.
2. Activity unit must match the factor unit.
3. Evidence reference is required.
4. Anomalies require review and are represented in lifecycle status.

Report lifecycle:

1. A report references an inventory, report type, period, compliance mapping,
   audit evidence, approval, and approver.
2. Missing approval, compliance mapping, audit evidence, or approver denies
   publishing.
3. Total carbon dioxide equivalent is calculated from the inventory activities.

Target lifecycle:

1. A target references an inventory, baseline year, target year, baseline
   carbon dioxide equivalent, and target reduction.
2. Missing baseline denies creation.
3. Current progress is calculated from inventory totals.

AI-agent lifecycle:

1. Agent is registered with runtime, role, scope, tenant, and disclosure.
2. Runtime must be one of `codex`, `claude_code`, `opencode`, or `pi`.
3. Role must be one of the configured ESGC roles.
4. Agent contributions are audit-visible and cannot bypass policy decisions.

## Rule Engine

Rules must deny or require review for:

- missing tenant context;
- missing inventory owner or boundary;
- unapproved factor source;
- missing factor source evidence or version;
- missing activity evidence;
- report without approval, compliance mapping, or audit evidence;
- target without baseline;
- emission anomaly without review;
- unregistered, unsupported, unscoped, or undisclosed AI agents;
- lifecycle state changes without audit evidence;
- batch ESG mutations that do not use Bytewax.

## UI And Theme

The APG Python UI contract exposes dashboard, emissions, factors, data sources,
reports, targets, agents, rules, audit, and settings routes. The theme uses
compact operational density with distinct treatments for emissions cards,
factor libraries, report evidence, target progress, ESGC-agent scope, stream
health, and audit evidence.

## Streaming

Batch ESG mutation must use Bytewax. The stream topic is `apg.esgc.lifecycle`,
and state covers inventories, factors, activities, reports, targets, ESGC
agents, and audit events. Live Bytewax topology deployment is an adapter
concern, but the package declares and enforces the guardrail.

## Adapter Boundaries

Adapters must handle:

- meter and source-system integrations;
- geospatial reporting boundaries through `geos`;
- durable audit evidence through `audl`;
- compliance framework mapping through `comp`;
- forecasting through `pred`;
- authentication and permission checks through `auth`;
- Bytewax lifecycle topology and operational monitoring.

## Acceptance Gates

- Contract validates through the APG capability registry.
- Configuration schema includes emissions, data sources, reporting, targets,
  ESGC agents, governance, observability, adapters, UI, and theme.
- Rules cover inventory, factor, activity, reporting, target, agent, audit, and
  Bytewax guardrails.
- Service can create inventories, register factors, record activities, publish
  reports, create targets, register agents, summarize state, and validate batch
  mutation streams.
- API helpers and view models expose the same lifecycle surfaces.
- Generated semantic evidence exposes provides/requires, routes, rules, theme,
  and streaming.
- README, specification, plan, progress log, focused tests, implementation
  audit, publish plan, and stale-marker scan are current.
