# FREC Capability Specification

## Purpose

FREC makes facial recognition a first-class APG capability. It must let generated applications compose face consent, template enrollment, liveness-backed verification, watchlist identification, emotion-analysis governance, review decisions, and audit evidence without binding generated apps to camera hardware, model servers, or production web frameworks.

## Functional Requirements

1. Record tenant-scoped face consent with subject, purpose, evidence, active status, and revocation.
2. Enroll face templates only when active consent exists, the template hash is present, template encryption evidence exists, and face quality meets threshold.
3. Record liveness evidence with score, spoof signal, deepfake signal, and capture context.
4. Verify a face only when tenant context, active consent, active template, matching subject, liveness evidence, and sufficient confidence are present.
5. Route low-confidence or low-quality decisions to review instead of silently accepting them.
6. Create watchlists only with policy, owner, and reason.
7. Add watchlist subjects only when the subject has an active template and consent.
8. Run identification only when an active watchlist policy is attached.
9. Govern watchlist hits and low-confidence identification matches through review.
10. Run emotion analysis only when an explicit approved purpose exists; aggregate-only mode remains the default.
11. Deny cross-tenant record access.
12. Emit audit evidence for lifecycle transitions.
13. Use Bytewax for batch recognition and watchlist mutation streams.

## Configuration

The contract must include configuration sections for consent, recognition, enrollment, templates, liveness, verification, identification, watchlists, emotion, privacy, reviews, security, governance, observability, adapters, UI, and theme.

Required adapters:

- `generated_app_runtime`: `face_runtime.FrecService`
- `helper_runtime`: `face_runtime.py`
- `api_helpers`: `api_helpers.py`
- `view_models`: `view_models.py`
- `production_runtime`: `service.py`
- `production_api`: `api.py`
- `production_views`: `views.py`
- `event_stream`: `bytewax`
- `biometric_processing`: `biop`
- `computer_vision`: `cvsn`
- `ai_core`: `aicr`
- `encryption`: `encr`
- `audit_sink`: `audl`
- `mfa_provider`: `mfau`
- `cache`: `cach`
- `metrics_sink`: `moni`

## Rule Engine

The rule engine must be deterministic and executable without external dependencies. It must cover tenant context, consent, enrollment quality, template encryption, liveness, spoof/deepfake detection, verification confidence, watchlist policy, identification threshold, emotion purpose, review independence, Bytewax streams, cross-tenant denial, and state-change audit.

## Runtime Contract

`FrecService` must expose:

- `describe`
- `evaluate`
- `record_face_consent`
- `revoke_face_consent`
- `enroll_face`
- `retire_template`
- `record_liveness`
- `verify_face`
- `create_watchlist`
- `add_watchlist_subject`
- `identify_face`
- `request_review`
- `decide_review`
- `analyze_emotion`
- list helpers, `dashboard_summary`, and `package`

The runtime stores deterministic metadata and decisions. It does not perform real face inference or raw-image processing.

## UI Contract

FREC must expose route metadata for dashboard, subjects, consents, enrollment, templates, verification, identification, liveness, watchlists, reviews, emotion, audit, and settings. Theme components must cover face quality, consent scope, template gallery, match gallery, liveness trace, watchlist table, review queue, emotion governance, and audit timeline.

## Acceptance Criteria

- The capability has root `README.md`, `SPECIFICATION.md`, and `PLAN.md`.
- The contract exposes at least 30 rules and at least 12 UI routes.
- The generated semantic model is derived from the live contract.
- `app.self_test()` fails when rule, route, adapter, or runtime evidence becomes stale.
- Focused tests cover positive lifecycle, key guardrails, API helpers, UI models, and package metadata.
