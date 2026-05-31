# HLTH Follow-Up Work

This file tracks remaining work after the current dependency-light HLTH
capability packet. The executable packet already covers component registration,
health checks, baselines, predictions, alerts, incidents, remediation review,
deployment gates, first-class health-agent registration, Bytewax lifecycle batch
validation, UI metadata, theme metadata, semantic-model evidence, and focused
tests.

## Runtime Adapters

- Bind active probe runners and service discovery to `HlthService`.
- Add MONI, OpenTelemetry, Kubernetes, cloud, and infrastructure feed adapters.
- Add notification, ticketing, incident, remediation, and deployment adapters
  that honor HLTH guardrails.
- Add AI-runtime adapters for Codex, Claude Code, opencode, Pi, and future
  agent providers without bypassing HLTH agent guardrails.

## Bytewax Topology

- Build the durable `hlth.lifecycle` Bytewax topology.
- Persist accepted and denied lifecycle batches outside in-memory service
  state.
- Add replay, backfill, checkpoint, and watermark validation.
- Add operational runbooks for lifecycle stream failures.

## Persistence and Retention

- Replace in-memory control-plane stores with tenant-scoped durable storage.
- Enforce configured health check, baseline, prediction, incident, remediation,
  deployment-gate, health-agent, lifecycle-batch, and audit retention.
- Add migration scripts for all dependency-light lifecycle records.

## UI Runtime

- Render generated screens in the selected APG shell.
- Add dashboard, component map, check timeline, baseline, prediction, alert,
  incident, remediation, deployment-gate, agent-roster, lifecycle-batch,
  adapter-health, audit, and settings browser checks.
- Verify tenant visual themes across compact and dense layouts.

## Verification

- Add integration tests once durable adapters exist.
- Add rendered UI checks once the shell route is available.
- Add load, failover, and migration checks after persistence and Bytewax
  topology are live.
- Keep battery-conscious focused tests for capability packet changes.
