# Anomaly Detection Capability Specification

- **Capability Name**: Anomaly Detection
- **Capability ID**: `anom`
- **Category**: common
- **Version**: 1.0.0

## Purpose

ANOM provides deterministic anomaly detection for APG monitoring, event, and
behavioral signals. It registers tenant-scoped monitoring sources, builds
statistical baselines from historical observations, scores new observations,
opens governed investigations for severe signals, records feedback, and exposes
view models for signal boards, baseline consoles, investigation queues, and
tuning review.

## Current Executable Governance Slice

The package includes a dependency-light `AnomService` runtime for generated APG
applications and capability composition. Monitoring vendors, incident tools,
alerting systems, workflow engines, and persistent stores remain adapter
boundaries while the local package executes deterministic detection governance.

Current package-backed lifecycle:

1. Register a tenant-owned monitoring source.
2. Build a baseline from sufficient history.
3. Score observations and emit anomaly signals with root-cause hints.
4. Require an investigation owner for critical signals.
5. Open investigations for owned anomalous signals.
6. Close investigations only with tenant match, actor, resolution, and
   resolution evidence.
7. Require tuning review when false-positive feedback crosses the configured
   threshold.
8. Keep all mutable runtime state tenant-qualified so duplicate IDs across
   tenants cannot collide.
9. Emit tenant-scoped audit events for source, baseline, signal, investigation,
   and feedback lifecycle changes.

Focused proof commands:

```bash
./.venv/bin/pytest -q capabilities/common/anom/test_capability_contract.py capabilities/common/anom/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/anom --json
./.venv/bin/apg capabilities publish-plan capabilities/common/anom --json
git diff --check -- capabilities/common/anom
```

## Provided Services

- `monitoring_source_registry`
- `baseline_profile_management`
- `metric_anomaly_detection`
- `event_anomaly_scoring`
- `investigation_queue`
- `feedback_tuning_loop`
- `anomaly_signal_view_models`

## Required Services

- `tenant_context`
- `pred`
- `aicr`
- `moni`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations. Baseline creation enforces the configured minimum historical
observations before detection can run.

## Rules

- `tenant_context_required`
- `detection_requires_monitoring_source`
- `baseline_requires_history`
- `critical_anomaly_requires_owner`
- `baseline_reset_requires_approval`
- `high_false_positive_rate_requires_tuning`

## Runtime Behavior

`service.py` owns dependency-light registries for monitoring sources, baselines,
observations, anomaly signals, investigations, feedback, and audit events.
`anomaly_engine.py` builds statistical baselines and scores observations with
sensitivity-specific thresholds. Critical signals require owners before they can
enter the investigation queue, closure requires evidence, and false-positive
feedback can force tuning review.

## UI

The package exposes 7 APG Python UI route contract(s) through `views.py` and the
package semantic model. View models cover the dashboard, signal board, baseline
console, investigation queue, and feedback review.

## Theme

The package uses the `anom_signal_console` APG theme contract for anomaly
severity cards, baseline drift charts, investigation timelines, and false
positive review meters.
