# Intelligence Dashboard Capability Specification

`intel_dashboard` is the APG Intelligence Dashboard capability. It composes
governed intelligence data into dashboards, sources, metrics, widgets, filters,
views, shares, reviews, UI models, Bytewax lifecycle events, and
provider-neutral AI-agent composition surfaces.

## Capability Summary

- Capability ID: `intel_dashboard`
- Display name: Intelligence Dashboard
- Target: Python executable capability package
- Event processor: Bytewax
- Event stream: `apg.intel.dashboard.lifecycle`
- Theme: `intel_dashboard_control`
- Agent runtimes: `codex`, `claude_code`, `opencode`, `pi`

## Composition Interfaces

The package provides authority, workspace, dashboard composition, source,
metric, widget, filter, view, share, review, and AI-agent workflows. It
requires APG auth, audit, notification, NLP, graph, RAG, and geospatial
capabilities so generated applications can compose dashboards with identity,
evidence, retrieval, graph context, map context, and downstream notifications.

## Runtime Shape

The service keeps tenant-scoped in-memory records for the executable baseline
while leaving rendered UI frameworks, live query engines, persistent layout
stores, graph/RAG writes, map rendering, notification delivery, and durable
Bytewax workers behind adapter boundaries.

## Governance

Every write path evaluates deterministic rules before mutation. The rules
require tenant context, policy attachment, lawful authority, evidence,
classification, owners, source custodians, metric evidence, approval for
shares, Bytewax routing, and human approval for privileged AI-agent scopes.
Uncited metrics, classification leaks, source tampering, privacy bypasses,
autonomous shares, and unapproved public views are denied.

