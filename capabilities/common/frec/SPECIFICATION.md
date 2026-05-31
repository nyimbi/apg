# FREC Capability Specification

## Purpose

FREC makes facial recognition a first-class APG capability. It must let generated applications compose face consent, template enrollment, liveness-backed verification, watchlist identification, emotion-analysis governance, review decisions, and audit evidence without binding generated apps to camera hardware, model servers, or production web frameworks.

The current packet also makes facial-recognition governance agents first-class citizens. Generated applications can register provider-neutral AI agents for consent review, enrollment review, liveness review, verification review, watchlist review, identification review, emotion-governance review, privacy review, lifecycle batch review, and stewardship. Agent runtimes such as `codex`, `claude_code`, `opencode`, and `pi` remain adapter-backed; FREC governs their declared role, scope, ownership, purpose, contribution disclosure, and human approval posture.

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
14. Register facial-recognition governance agents as tenant-scoped records with supported runtime, supported role, explicit scope, accountable owner, declared purpose, and machine-contribution disclosure.
15. Mark privileged facial-recognition agent roles as pending review unless human approval evidence is recorded.
16. Validate lifecycle batches only when they are non-empty, declare a supported FREC operation, and are routed through Bytewax.

## Configuration

The contract must include configuration sections for consent, recognition, enrollment, templates, liveness, verification, identification, watchlists, emotion, privacy, reviews, security, governance, observability, adapters, UI, and theme.

The contract must also include:

- `agents`: first-class provider-neutral facial-recognition governance agents with supported runtimes `codex`, `claude_code`, `opencode`, and `pi`; supported roles; privileged roles; required owner, purpose, scope, contribution disclosure, and privileged-role human approval.
- `streaming`: Bytewax lifecycle metadata with `frec.lifecycle`, `event_time` watermarking, Bytewax as required processor, FREC lifecycle batch operations, FREC topics, and no broker-core/broker-specific queue dependency.

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
- `agent_adapter`: `aicr_provider_neutral_facial_recognition_agent_adapter`

## Rule Engine

The rule engine must be deterministic and executable without external dependencies. It must cover tenant context, consent, enrollment quality, template encryption, liveness, spoof/deepfake detection, verification confidence, watchlist policy, identification threshold, emotion purpose, review independence, Bytewax streams, cross-tenant denial, and state-change audit.

It must also cover unsupported agent runtimes, unsupported agent roles, missing agent scope, missing owner, missing purpose, missing machine-contribution disclosure, privileged agent registration without human approval, empty lifecycle batches, unsupported lifecycle operations, and non-Bytewax lifecycle streams.

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
- `register_facial_recognition_agent`
- `validate_frec_lifecycle_batch`
- list helpers, `dashboard_summary`, and `package`

The runtime stores deterministic metadata and decisions. It does not perform real face inference or raw-image processing.

## UI Contract

FREC must expose route metadata for dashboard, subjects, consents, enrollment, templates, verification, identification, liveness, watchlists, reviews, emotion, agents, lifecycle, audit, and settings. Theme components must cover face quality, consent scope, template gallery, match gallery, liveness trace, watchlist table, review queue, emotion governance, facial-recognition agent roster, Bytewax lifecycle panel, and audit timeline.

## Acceptance Criteria

- The capability has root `README.md`, `SPECIFICATION.md`, and `PLAN.md`.
- The contract exposes at least 44 rules and at least 15 UI routes.
- The contract, app semantic model, release report, and manifest include first-class facial-recognition governance agents and Bytewax lifecycle metadata.
- The generated semantic model is derived from the live contract.
- `app.self_test()` fails when rule, route, adapter, or runtime evidence becomes stale.
- Focused tests cover positive lifecycle, key guardrails, agent registration, lifecycle batch validation, API helpers, UI models, and package metadata.
