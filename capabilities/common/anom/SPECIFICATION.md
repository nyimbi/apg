# ANOM Capability Specification

## Identity

- Capability ID: `anom`
- Display name: Anomaly Detection
- Category: `common`
- Owner: APG Platform Team
- Runtime shell: `apg_python`
- Theme: `anom_signal_console`

## Purpose

ANOM provides tenant-scoped anomaly detection for metrics, events, and monitored
signals. It registers monitoring sources, builds baselines, scores observations,
routes severe anomalies into investigations, records feedback, and emits
governance evidence for detection and investigation decisions.

The package must remain dependency-light. Statistical scoring, state
transitions, rules, API helpers, and view models must be executable without
external monitoring systems, vector stores, alerting systems, or incident tools.
Those systems remain adapter boundaries.

## Users And Outcomes

- Operators can register monitored sources and build baselines from historical
  observations.
- SRE and risk teams can detect anomalous values and receive root-cause hints.
- Investigation owners can triage and close severe anomaly investigations with
  resolution evidence.
- Platform teams can review false-positive feedback and tune detection policy.
- Generated APG applications can compose ANOM with MONI, PRED, AICR, AUDL,
  WFLO, NTFY, and HLTH without coupling to one monitoring vendor.

## Domain Model

ANOM owns these package-level records:

- `MonitoringSource`: tenant-owned metric, stream, or event source.
- `BaselineProfile`: statistical baseline with sensitivity and history count.
- `Observation`: single measured value or event score.
- `AnomalySignal`: scored signal with severity, status, and root-cause hints.
- `Investigation`: governed investigation assigned to an anomaly signal.
- `DetectionFeedback`: reviewer label used to tune detection quality.
- `AnomalyAuditEvent`: tenant-scoped evidence event for source, baseline,
  detection, investigation, and feedback lifecycle changes.

All mutable runtime state must be tenant-qualified so duplicate IDs in different
tenants cannot overwrite or expose each other.

## Lifecycle

The focused lifecycle is:

1. Register a tenant-owned monitoring source.
2. Build a baseline from sufficient historical values.
3. Detect a new observation against the baseline.
4. Require an owner for critical anomaly signals.
5. Open an investigation for owned critical or high signals.
6. Close the investigation only with tenant match, actor, resolution, and
   evidence.
7. Record feedback and require tuning review when the false-positive rate is
   too high.
8. Emit audit events for each important lifecycle decision.

## Rules And Guardrails

The contract rules are executable guardrails:

- `tenant_context_required`: operations require tenant context.
- `detection_requires_monitoring_source`: detection requires a monitoring
  source.
- `baseline_requires_history`: baseline creation requires enough observations.
- `critical_anomaly_requires_owner`: critical signals require an owner.
- `baseline_reset_requires_approval`: baseline reset requires approval.
- `high_false_positive_rate_requires_tuning`: high false-positive rates require
  tuning review.

Service methods must enforce these rules and expose the same decisions through
API helpers and view models.

## UI And Theme

ANOM exposes route and view-model surfaces for:

- dashboard summary;
- signal board;
- baseline console;
- investigation queue;
- rule management;
- feedback review;
- settings.

The `anom_signal_console` theme must provide semantic tokens and component
metadata for severity, baseline drift, investigation ownership, and
false-positive review.

## Adapter Boundaries

These integrations remain replaceable:

- monitoring backends and metric stores;
- alerting and incident-management systems;
- workflow engines for escalation;
- predictive-model and AI scoring engines;
- audit and SIEM exporters;
- persistent storage providers.

Local package tests must not require those systems.

## Acceptance Gates

Focused ANOM proof:

```bash
./.venv/bin/pytest -q capabilities/common/anom/test_capability_contract.py capabilities/common/anom/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/anom --json
./.venv/bin/apg capabilities publish-plan capabilities/common/anom --json
git diff --check -- capabilities/common/anom
```
