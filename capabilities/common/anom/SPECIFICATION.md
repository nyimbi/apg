# ANOM Specification

## Purpose

ANOM provides APG with first-class anomaly detection for generated
applications. It supports monitored source registration, baseline management,
observation scoring, signal triage, investigation workflows, feedback tuning,
UI composition, and auditable policy enforcement.

## Scope

This packet establishes the executable baseline for ANOM:

- Contract-driven configuration, schema, adapters, deterministic rules, UI
  routes, and visual theme tokens.
- A dependency-light runtime service for generated applications.
- UI view models that can be composed into APG screens.
- Package evidence that can be published and self-tested from the current
  executable contract.
- Focused tests for the contract, lifecycle, guardrails, view models, and
  package evidence.

## Actors

- Operator: registers sources, creates baselines, and monitors signal boards.
- SRE or risk owner: investigates critical and high anomaly signals.
- Detection steward: records feedback and tunes false-positive behavior.
- Platform operator: configures Bytewax streams, monitoring, notifications,
  audit, auth, metrics, cache, and generated app deployment.

## Functional Requirements

### Source Lifecycle

- Register monitoring sources with tenant, name, kind, owner, and labels.
- Deny missing tenant, name, owner, or kind.
- Require review for unknown source kinds.

### Baseline Lifecycle

- Create baselines from tenant-scoped sources, metric names, historical values,
  and sensitivity.
- Deny missing source, metric, sufficient history, or sensitivity.
- Require review for unknown sensitivity values.
- Reset baselines only with approval evidence.

### Detection Lifecycle

- Score observations against registered baselines using deterministic z-score
  thresholds and sensitivity.
- Deny missing source, baseline, metric, or observed value.
- Deny critical anomaly acceptance without an investigation owner.
- Require triage review for high-severity anomalies without evidence.
- Deny cross-tenant source or baseline use.

### Investigation Lifecycle

- Open investigations from tenant-scoped anomaly signals and owners.
- Close investigations only with resolution, closing actor, and resolution
  evidence.
- Emit audit events for source, baseline, signal, investigation, and feedback
  lifecycle changes.

### Feedback Lifecycle

- Record feedback against anomaly signals with reviewer, label, and notes.
- Deny missing signal, reviewer, or label.
- Require review for unknown labels.
- Require tuning review when false-positive rate exceeds the configured
  threshold.

### UI and Theming

- Expose routes for dashboard, sources, baselines, detector, signals,
  investigations, alerts, rules, feedback, quality, audit, and settings.
- Provide route-specific view models.
- Publish signal-console theme tokens and component hints.

### Adapters

- Use Bytewax for batch anomaly/event streams.
- Expose adapter keys for generated runtime, helper runtime, HTTP API, PRED,
  AICR, MONI, WFLO, NTFY, HLTH, CONF, AUTH, AUDL, MONI metrics, and CACH.

## Non-Goals

- Live monitoring backend ingestion.
- Live Bytewax stream execution.
- Live notification or incident-management dispatch.
- Persistent database migrations.
- Browser-rendered UI validation.
- Load, latency, precision/recall, drift, and throughput benchmarking.

These are later integration and hardening tasks once the executable baseline is
stable.

## Acceptance Criteria

- `get_capability_contract()` exposes at least 30 deterministic rules, at least
  12 UI routes, Bytewax adapter evidence, runtime adapter evidence, and theme
  component metadata.
- `AnomService` executes source, baseline, detection, investigation, feedback,
  reset, list, summary, audit, and APG record compatibility flows.
- Guardrail tests prove denied or review-required cases fail before invalid
  state is accepted.
- `app.self_test()` passes and fails if route, rule, Bytewax, or runtime
  evidence becomes stale.
- Package JSON evidence can be regenerated from `app.semantic_model()` and
  `app.component_manifest()`.
