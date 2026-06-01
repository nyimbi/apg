# Predictive Intelligence Capability Specification

`intel_prediction` is the APG Predictive Intelligence capability. It turns
lawful analytical intent into governed prediction workspaces, scenarios,
indicators, validated models, forecasts, risk projections, early warnings,
recommendations, reviews, UI models, Bytewax lifecycle events, and AI-agent
composition surfaces.

## Capability Summary

- Capability ID: `intel_prediction`
- Display name: Predictive Intelligence
- Target: Python executable capability package
- Event processor: Bytewax
- Event stream: `apg.intel.prediction.lifecycle`
- Theme: `intel_prediction_control`
- Agent runtimes: `codex`, `claude_code`, `opencode`, `pi`

## Composition Interfaces

The package provides workflows for authority, workspace, scenario, indicator,
model, forecast, projection, warning, recommendation, review, and AI-agent
operations. It requires APG auth, audit, notification, NLP, graph, RAG, and
geospatial capabilities so generated applications can compose prediction with
identity, evidence, enrichment, graph context, retrieval, map context, and
downstream alerting.

## Runtime Shape

The service is intentionally deterministic and adapter-friendly. It keeps
tenant-scoped in-memory records for the executable package baseline while
leaving persistent storage, live forecasting engines, feature stores, model
registries, dashboard rendering, graph writes, RAG indexing, notification
delivery, and durable Bytewax workers behind explicit adapter boundaries.

## Governance

Every write path evaluates deterministic rules before mutation. The rules
require tenant context, policy attachment, lawful authority, evidence,
classification, analyst ownership, validation, approval for warnings and
recommendations, Bytewax routing for lifecycle batches, and human approval for
privileged AI-agent scopes. Unsafe scopes such as unsupported automated
decisions, hallucinated forecasts, privacy bypasses, unapproved model
deployment, autonomous warnings, and autonomous recommendations are denied.

