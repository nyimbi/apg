# Intelligence Dashboard Specification

## Purpose

Intelligence Dashboard lets APG applications present operational intelligence
in governed, composable dashboards. It is designed for analysts, supervisors,
executives, partner users, field teams, and watch centers that need dense,
trustworthy, evidence-backed dashboards rather than unmanaged charts.

## Users

- Analysts who compose dashboards and curate widgets.
- Metric stewards who attach data sources and evidence.
- Supervisors who review access, shares, and public views.
- Operations teams that consume current intelligence and watch-center status.
- AI-agent supervisors who delegate bounded layout, metric, source, access,
  theme, and briefing-preparation work.

## Functional Scope

- Authorities: lawful dashboard mandates with scope, classification, approver,
  expiry, and evidence.
- Workspaces: governed containers for operations centers, threat watch,
  executive overview, investigation rooms, incident rooms, partner views, and
  field dashboards.
- Dashboards: operational, strategic, threat, incident, investigative,
  executive, and partner boards with owner and classification.
- Data sources: capability summaries, graph queries, RAG extracts, geospatial
  layers, alert feeds, reporting products, prediction projections, and threat
  assessments.
- Metrics: counts, rates, risk scores, trends, statuses, coverage, latency, and
  confidence indicators.
- Widgets: KPI tiles, trend charts, maps, network graphs, tables, timelines,
  watchlists, and status boards.
- Filters: time range, classification, geography, source, risk level, owner,
  and status controls.
- Views: analyst, supervisor, executive, partner, field, and public-safety
  views.
- Shares: approved dashboard sharing to internal, partner, executive, field,
  watch-center, or case-team recipients.
- Reviews: human review outcomes for lifecycle artifacts.
- AI agents: provider-neutral runtimes with bounded roles and explicit scope.

## Out Of Scope

This package does not render a live browser UI, execute live queries, mutate
graph/RAG stores, render maps, send notifications, persist layouts, or run
durable stream topologies. Those remain adapter responsibilities until their
contracts are explicit.

## Lifecycle

1. Record authority.
2. Create dashboard workspace.
3. Record dashboard.
4. Register governed data source.
5. Record metric.
6. Record widget.
7. Record filter.
8. Record view.
9. Record approved share.
10. Record human review.
11. Register bounded AI agents.
12. Route lifecycle batches through Bytewax.

## Rule Engine

The deterministic rule engine denies missing tenant context, unsupported
taxonomy values, missing evidence, missing authority, missing owners, missing
source custodians, invalid confidence, missing share approval, non-Bytewax
batches, unsupported agent runtimes or roles, missing agent scope, privileged
agent actions without approval, uncited metrics, classification leaks, source
tampering, privacy bypasses, autonomous shares, and unapproved public views.

## UI And Theme

The capability exposes APG Python UI route metadata for dashboard,
authorities, workspaces, dashboards, sources, metrics, widgets, filters, views,
shares, reviews, agents, and settings. The theme uses compact, work-focused
tokens under `intel_dashboard_control`.

## Adapter Boundaries

Generated applications compose this capability with auth, audit, notification,
NLP, graph, RAG, and geospatial capabilities. Production integrations should
bind rendered UI, live query engines, layout persistence, graph/RAG writes, map
rendering, notifications, and durable Bytewax workers through adapters without
bypassing this package's deterministic rules.

