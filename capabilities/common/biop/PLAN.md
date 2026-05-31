# BIOP Implementation Plan

## Scope

Implement one coherent dependency-light biometric governance lifecycle packet:

1. Consent recording and revocation.
2. Encrypted template enrollment and retirement.
3. Liveness-backed verification.
4. Cross-border privacy review.
5. Low-confidence match review.
6. First-class biometric governance-agent composition.
7. Bytewax-first biometric lifecycle batch validation.
8. API helpers, view models, contract routes/rules/theme evidence, package semantic evidence, and focused tests.

## Non-Goals

- Do not replace the existing production-oriented biometric service.
- Do not add new dependencies.
- Do not build sensor integrations, model inference, HSM encryption, or production web routes.
- Do not build runtime-specific Codex, Claude Code, opencode, or Pi clients; keep them behind AICR adapters.
- Do not introduce Kafka or broker-core lifecycle dependencies; BIOP lifecycle batches are Bytewax-first.
- Do not run full repository test suites while battery is constrained.

## Implementation Steps

### 1. Add Dependency-Light Runtime

Create `biometric_runtime.py` with dataclass records and a `BiopService` facade:

- `BiometricConsent`
- `BiometricTemplateRecord`
- `BiometricVerificationRecord`
- `BiometricReviewApproval`
- `BiometricAuditEvent`
- `BiometricAgentRecord`
- `BiopLifecycleBatchRecord`

Use tenant-qualified in-memory stores and deterministic validation. Keep all IDs tenant-local.

### 2. Add API Helpers and View Models

Create package helpers that generated APG applications can call directly:

- record/revoke consent;
- enroll/retire template;
- request verification;
- request/decide privacy review;
- request/decide match review;
- register/list biometric governance agents;
- validate/list Bytewax lifecycle batches;
- list all lifecycle records;
- dashboard, consent, template, verification, review, and audit view models.

### 3. Extend Capability Contract

Update `capability_contract.py`:

- add consent, template-vault, match-review, and privacy-review routes;
- add independent reviewer, active consent, and active template rules;
- add first-class agent and Bytewax lifecycle batch configuration;
- add agent and lifecycle routes;
- add deterministic agent registration and lifecycle batch rules;
- add theme components for consent center, review queues, privacy posture, agent roster, and lifecycle posture.

### 4. Refresh Package Evidence

Update `app.py` to derive the semantic model from the live contract instead of embedded stale JSON. Refresh:

- `semantic_model.json`
- `release_report.json`
- `package_manifest.json`
- `cap_spec.md`

### 5. Add Focused Tests

Update package tests to cover:

- contract shape and new routes/rules/theme;
- positive consent-template-verification-review-agent-lifecycle-audit lifecycle;
- negative missing consent, revoked consent, unencrypted template, low quality, low liveness, low confidence, self-review, duplicate review, stale review, missing cross-border review, rejected privacy review, retired template, and tenant isolation;
- negative unsupported agent runtime, privileged agent pending review, empty lifecycle batch, unsupported lifecycle operation, and non-Bytewax lifecycle stream;
- API-helper and view-model shared-state behavior;
- renamed package contract test path.

### 6. Review and Verification

Run focused proof only:

- `py_compile` on changed BIOP package files;
- focused BIOP package tests;
- `apg capabilities implementation-audit --root capabilities/common/biop --json`;
- `apg capabilities publish-plan capabilities/common/biop --json`;
- stale-marker search;
- `git diff --check`.

Review the diff manually and fix blocking lifecycle/guardrail gaps before commit.
