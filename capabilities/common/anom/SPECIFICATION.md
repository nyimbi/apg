# ANOM Specification

## Purpose

ANOM provides APG with first-class anomaly detection for generated
applications. It supports monitored source registration, baseline management,
observation scoring, signal triage, investigation workflows, feedback tuning,
first-class AI anomaly-agent composition, Bytewax lifecycle batch validation,
UI composition, and auditable policy enforcement.

## Scope

This packet establishes the executable baseline for ANOM:

- Contract-driven configuration, schema, adapters, deterministic rules, UI
  routes, and visual theme tokens.
- A dependency-light runtime service for generated applications.
- UI view models that can be composed into APG screens.
- Provider-neutral AI anomaly agents as executable state with runtime, role,
  scope, owner, purpose, disclosure, human-review status, and audit evidence.
- Bytewax-only lifecycle batch validation for anomaly mutations.
- Durable review-required source, baseline, signal, and feedback outcomes as
  pending-review records with matched rules and review reasons.
- Package evidence that can be published and self-tested from the current
  executable contract.
- Focused tests for the contract, lifecycle, guardrails, view models, and
  package evidence.

## Actors

- Operator: registers sources, creates baselines, and monitors signal boards.
- SRE or risk owner: investigates critical and high anomaly signals.
- Detection steward: records feedback and tunes false-positive behavior.
- AI anomaly agent: assists with source, baseline, detector, signal,
  investigation, feedback, alert, and baseline-reset review while remaining
  provider-neutral.
- Platform operator: configures Bytewax streams, monitoring, notifications,
  audit, auth, metrics, cache, and generated app deployment.

## Functional Requirements

### Source Lifecycle

- Register monitoring sources with tenant, name, kind, owner, and labels.
- Deny missing tenant, name, owner, or kind.
- Persist unknown source kinds as pending-review sources with deterministic
  matched-rule and reason evidence.

### Baseline Lifecycle

- Create baselines from tenant-scoped sources, metric names, historical values,
  and sensitivity.
- Deny missing source, metric, sufficient history, or sensitivity.
- Persist unknown sensitivity values as pending-review baselines with
  deterministic matched-rule and reason evidence.
- Reset baselines only with approval evidence.

### Detection Lifecycle

- Score observations against registered baselines using deterministic z-score
  thresholds and sensitivity.
- Deny missing source, baseline, metric, or observed value.
- Deny critical anomaly acceptance without an investigation owner.
- Persist high-severity anomalies without triage evidence as pending-review
  signals instead of opening investigations automatically.
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
- Persist unknown labels as pending-review feedback.
- Persist feedback as pending-review when false-positive rate exceeds the
  configured threshold and tuning review has not been recorded.

### AI Agent Lifecycle

- Register anomaly agents with tenant, name, runtime, role, scope, owner,
  purpose, machine-contribution disclosure, and human-approval metadata.
- Support provider-neutral runtimes `codex`, `claude_code`, `opencode`, and
  `pi`.
- Support roles for source review, baseline review, detector review,
  signal-triage review, investigation review, feedback-tuning review,
  alert-dispatch review, baseline-reset review, and anomaly stewardship.
- Deny unsupported runtimes, unsupported roles, missing scope, missing owner,
  missing purpose, and missing machine-contribution disclosure.
- Put privileged anomaly-agent roles into pending review when human approval
  evidence is absent.
- Keep live agent invocation, credentials, and provider-specific routing behind
  the AICR adapter boundary.

### Bytewax Lifecycle Batches

- Validate ANOM lifecycle mutation batches through the declared Bytewax stream
  contract.
- Accept only configured lifecycle operations: source, baseline, detection,
  signal, investigation, feedback, alert, and anomaly-agent batches.
- Deny non-Bytewax lifecycle streams while preserving denied-batch evidence for
  audit and UI review.

### UI and Theming

- Expose routes for dashboard, sources, baselines, detector, signals,
  investigations, alerts, rules, feedback, quality, agents, lifecycle batches,
  audit, and settings.
- Provide route-specific view models.
- Expose pending-review queues for source, baseline, signal, feedback, and
  anomaly-agent governance.
- Publish signal-console theme tokens and component hints.

### Adapters

- Use Bytewax for batch anomaly/event streams.
- Expose adapter keys for generated runtime, helper runtime, HTTP API, PRED,
  AICR, MONI, WFLO, NTFY, HLTH, CONF, AUTH, AUDL, MONI metrics, and CACH.

## Non-Goals

- Live monitoring backend ingestion.
- Live Bytewax stream execution.
- Live AI-agent CLI/API invocation.
- Live notification or incident-management dispatch.
- Persistent database migrations.
- Browser-rendered UI validation.
- Load, latency, precision/recall, drift, and throughput benchmarking.

These are later integration and hardening tasks once the executable baseline is
stable.

## Acceptance Criteria

- `get_capability_contract()` exposes at least 39 deterministic rules, at least
  14 UI routes, first-class agent metadata, Bytewax lifecycle metadata, runtime
  adapter evidence, and theme component metadata.
- `AnomService` executes source, baseline, detection, investigation, feedback,
  reset, anomaly-agent, lifecycle-batch, list, summary, audit, and APG record
  compatibility flows.
- Guardrail tests prove denied cases fail before invalid state is accepted, and
  review-required cases persist as pending-review records with matched rules and
  review reasons.
- `app.self_test()` passes and fails if route, rule, Bytewax, or runtime
  evidence becomes stale.
- Package JSON evidence can be regenerated from `app.semantic_model()` and
  `app.component_manifest()`.
