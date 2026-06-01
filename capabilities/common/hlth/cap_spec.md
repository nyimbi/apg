# HLTH Capability Summary

HLTH provides APG applications with a tenant-scoped health checks and
diagnostics control plane. It focuses on executable composition contracts:
component registration, health check records, baselines, predictions, alerts,
incidents, remediation review, deployment gates, rules, generated-application UI
metadata, health-agent composition, Bytewax lifecycle validation, and adapter
boundaries. Review-required and denied records preserve durable policy evidence
for generated health consoles and audit timelines.

## Current Executable Scope

- Register tenant-scoped components with owner, type, environment,
  criticality, dependency, and status metadata.
- Record health checks with deterministic rule decisions and audit evidence.
- Create baseline records and prediction records for generated workflows.
- Open alerts and incidents from critical checks.
- Request and decide remediation with runbook, production approval,
  independent reviewer, and notes evidence.
- Evaluate deployment gates against unresolved critical incidents.
- Register first-class health agents with runtime, role, scope, owner, purpose,
  contribution-disclosure, and human-approval evidence; otherwise valid
  privileged agents without approval are retained as `pending_review`.
- Validate lifecycle mutation batches through a Bytewax-first stream contract
  and persist denied non-Bytewax batch evidence before raising.
- Expose pending-review queues and policy fields for checks, predictions,
  alerts, incidents, remediation requests, deployment gates, health agents, and
  lifecycle batches.
- Publish deterministic rules, UI routes, theme components, semantic model, and
  release evidence.

## Adapter Scope

The following remain runtime adapters rather than hard dependencies of the
dependency-light package:

- active probe runners and service discovery
- OpenTelemetry, MONI, Prometheus, Kubernetes, cloud, and infrastructure feeds
- external notification, ticketing, incident, and remediation systems
- ML model training and live prediction engines
- production persistence, retention, and compliance stores
- rendered dashboards and browser runtime shells
- Codex, Claude Code, opencode, Pi, and future AI runtime clients
- durable Bytewax topologies and stream processors

Adapters must not bypass tenant context, component registration, alert route,
incident ownership, baseline review, prediction confidence, remediation review,
deployment gate rules, health-agent guardrails, or Bytewax lifecycle batch
validation. They must also preserve `policy_decision`, `matched_rules`,
`review_reasons`, and `review_evidence` when syncing HLTH records into external
stores.

## Proof Commands

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/hlth/capability_contract.py \
  capabilities/common/hlth/service.py \
  capabilities/common/hlth/api.py \
  capabilities/common/hlth/view_models.py \
  capabilities/common/hlth/app.py

./.venv/bin/pytest -q \
  capabilities/common/hlth/test_capability_contract.py \
  capabilities/common/hlth/tests/test_package_contract.py

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/hlth --json
./.venv/bin/apg capabilities publish-plan capabilities/common/hlth --json
```
