# MONI Follow-Up Work

This file tracks remaining work after the current dependency-light MONI
capability packet. The executable packet already covers source registration,
signal governance, SLOs, alerts, incidents, remediation review, first-class
monitoring-agent registration, Bytewax lifecycle batch validation, UI metadata,
theme metadata, semantic-model evidence, and focused tests.

## Runtime Adapters

- Bind OpenTelemetry collector ingestion to `MoniService.ingest_signal`.
- Add production metrics, log, and trace store adapters.
- Add notification and incident-tool adapters for critical alert routing.
- Add SIEM/SOAR and runbook-executor adapters that honor remediation approval
  decisions.
- Add AI-runtime adapters for Codex, Claude Code, opencode, Pi, and future
  agent providers without bypassing MONI agent guardrails.

## Bytewax Topology

- Build the durable `moni.lifecycle` Bytewax topology.
- Persist accepted and denied lifecycle batches outside in-memory service
  state.
- Add replay, backfill, checkpoint, and watermark validation.
- Add operational runbooks for lifecycle stream failures.

## Persistence and Retention

- Replace in-memory control-plane stores with tenant-scoped durable storage.
- Enforce configured metrics, logs, traces, and compliance-evidence retention.
- Add migration scripts for source, signal, SLO, alert, incident, remediation,
  monitoring-agent, lifecycle-batch, and audit records.

## UI Runtime

- Render generated screens in the selected APG shell.
- Add dashboard, source inventory, signal explorer, SLO, alert, incident,
  remediation, agent-roster, lifecycle-batch, adapter-health, audit, and
  settings browser checks.
- Verify tenant visual themes across compact and dense layouts.

## Verification

- Add integration tests once durable adapters exist.
- Add rendered UI checks once the shell route is available.
- Add load, failover, and migration checks after persistence and Bytewax
  topology are live.
- Keep battery-conscious focused tests for capability packet changes.
