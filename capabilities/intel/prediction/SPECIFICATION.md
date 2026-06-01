# Predictive Intelligence Specification

## Purpose

Predictive Intelligence lets an APG application define, govern, and execute
predictive analytical workflows. It is designed for analysts and operators who
need to reason about likely events, changing risks, emerging threats, fraud
patterns, operational demand, public-safety signals, or strategic scenarios
while keeping the full chain of authority, evidence, validation, review, and
approval visible.

## Users

- Intelligence analysts who define scenarios, indicators, and forecasts.
- Model stewards who register validated predictive models.
- Reviewers and approvers who authorize warnings and recommendations.
- Operations teams that consume projections and early warnings.
- AI-agent supervisors who delegate bounded support work to provider-neutral
  agent runtimes.

## Functional Scope

The capability supports these lifecycle objects:

- Authorities: lawful prediction mandates with scope, classification, approver,
  expiry, and evidence.
- Workspaces: governed analytical containers for prediction work.
- Scenarios: concrete predictive questions tied to workspaces, owners, horizons,
  and evidence.
- Indicators: leading, lagging, anomaly, behavioral, geospatial, network, or
  text signals used to support predictions.
- Models: rulesets, statistical models, machine-learning models, simulations,
  graph forecasts, geospatial forecasts, and NLP forecasts with validation
  references and risk levels.
- Forecasts: probability, trend, scenario outcome, event likelihood, impact, or
  risk forecasts tied to validated models.
- Projections: risk, impact, timeline, resource, threat, fraud, or confidence
  projections derived from forecasts.
- Warnings: approved early-warning or threshold notices with trigger evidence.
- Recommendations: approved action proposals such as monitor, investigate,
  mitigate, escalate, request collection, review model, or close.
- Reviews: human review outcomes for lifecycle records.
- AI agents: `codex`, `claude_code`, `opencode`, and `pi` runtimes assigned to
  bounded prediction roles.

## Out Of Scope

The executable package does not run live model training, simulation engines,
feature stores, model registries, visualization renderers, graph mutations,
retrieval indexing, notification delivery, durable stream topologies, or
production persistence. Those remain adapter responsibilities until APG defines
their contracts.

## Lifecycle

1. Record authority.
2. Create prediction workspace.
3. Define scenario and horizon.
4. Attach indicators and evidence.
5. Register validated model.
6. Record forecast.
7. Record projection.
8. Approve and record warning when needed.
9. Approve and record recommendation when needed.
10. Record human review.
11. Register bounded AI agents.
12. Route lifecycle batches through Bytewax.

## Rule Engine

The rule engine is deterministic. It denies missing tenant context, unsupported
taxonomy values, missing evidence, missing lawful authority, invalid confidence
or probability scores, missing validation, missing analysts, missing approval,
non-Bytewax batch routing, unsupported agent runtimes or roles, privileged
agent actions without approval, unsupported automated decisions, hallucinated
forecasts, privacy bypasses, unapproved model deployment, autonomous warnings,
and autonomous recommendations.

## UI And Theme

The capability exposes APG Python UI route metadata for dashboard, authorities,
workspaces, scenarios, indicators, models, forecasts, projections, warnings,
recommendations, reviews, agents, and settings. The theme uses compact,
work-focused tokens under `intel_prediction_control` and component metadata
for the generated screens.

## Adapter Boundaries

Generated applications compose this capability with auth, audit, notification,
NLP, graph, RAG, and geospatial capabilities. Production integrations should
bind storage, model execution, feature stores, graph writes, RAG indexing,
notification delivery, rendered UI, and durable Bytewax workers through
adapters without bypassing the deterministic rules in this package.

