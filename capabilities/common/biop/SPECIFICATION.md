# BIOP Capability Specification

## Purpose

BIOP provides governed biometric processing for generated APG applications. It must let applications enroll biometric templates, authenticate subjects, run liveness checks, route sensitive decisions for review, and prove privacy/compliance controls without forcing every generated application to import the production Flask, Flask-AppBuilder, database, CV, voice, or biometric engine stack.

The package-level runtime is intentionally dependency-light. Production deployments may bind the same contract to hardware sensors, biometric matchers, HSM-backed template vaults, privacy engines, fraud systems, and model services through adapters. The package itself must remain executable, tenant-scoped, auditable, and fail-closed.

## Composable Capabilities

BIOP exposes these composable application components:

- Biometric consent center for explicit subject consent, modality scope, jurisdiction scope, evidence capture, and revocation.
- Encrypted template enrollment for face, voice, fingerprint, iris, palm, behavioral, and document-derived modalities.
- Liveness-backed verification for authentication and identity proofing.
- Low-confidence match review queue with independent reviewer decision and notes.
- Cross-border biometric processing privacy review queue with independent reviewer decision and notes.
- Template vault, retirement, and revocation state for privacy and security operations.
- First-class biometric governance agents for consent, enrollment, template-vault, liveness, match, privacy, retention, and lifecycle-batch review.
- Bytewax-first lifecycle batch validation for generated biometric mutation streams without Kafka or broker-core coupling.
- Audit event stream for consent, enrollment, verification, review, revocation, and retirement evidence.
- Dashboard and workbench view models that generated applications can render directly.

## Lifecycle

### 1. Consent Recording

Applications record a tenant-scoped biometric consent before any biometric processing. A valid consent includes:

- tenant ID;
- subject ID;
- consent purpose;
- allowed modalities;
- allowed jurisdictions;
- actor who captured consent;
- evidence reference;
- active status.

BIOP rejects biometric processing when consent is missing, revoked, scoped to the wrong subject, scoped to the wrong modality, or scoped to the wrong jurisdiction.

### 2. Template Enrollment

Applications enroll templates only after active consent is present. Enrollment requires:

- tenant context;
- active consent;
- modality in consent scope;
- encrypted template evidence;
- template hash;
- quality score at or above the configured threshold;
- retention policy.

BIOP stores only template metadata and encrypted-template evidence in the package runtime. Raw samples are not retained. Template IDs are tenant-local and do not collide across tenants.

### 3. Verification Request

A verification request binds a subject, modality, template, liveness score, match confidence, requester, and jurisdiction context. BIOP evaluates deterministic rules:

- missing tenant context denies;
- missing consent denies;
- missing or inactive template denies;
- missing liveness denies;
- low liveness score denies;
- cross-border processing without an approved privacy review denies;
- low confidence produces a pending human-review decision;
- sufficient confidence and liveness verifies automatically.

Caller-supplied booleans such as `human_review_recorded` or `privacy_review_recorded` are not accepted as governance evidence. Reviews must exist as matching, approved BIOP review records.

### 4. Privacy Review

Cross-border biometric processing requires a privacy review before verification can complete. The review must:

- match the tenant and verification;
- be requested by the operator who needs the processing;
- be decided by an independent reviewer;
- include reviewer notes;
- be approved before verification can proceed.

Rejected privacy reviews keep the verification rejected.

### 5. Match Review

Low-confidence matches require independent review. The review must:

- match the tenant and verification;
- be requested by the verification requester;
- be decided by an independent reviewer;
- include reviewer notes;
- either approve the verification or reject it.

BIOP prevents duplicate pending reviews for the same verification and prevents stale review decisions from mutating already-decided verifications.

### 6. Consent Revocation and Template Retirement

Consent revocation immediately blocks new processing under that consent and retires active templates bound to it. Templates may also be retired explicitly by an authorized actor with a reason. Verification requests against retired templates fail.

### 7. Audit Evidence

Every lifecycle transition emits tenant-scoped audit evidence with:

- event ID;
- event type;
- actor;
- subject ID;
- decision;
- reasons;
- metadata;
- timestamp.

Audit events are the package-level proof for generated applications. Production deployments may forward the same events to AUDL.

### 8. Biometric Governance Agent Composition

BIOP treats AI agents as first-class biometric governance citizens. A biometric agent registration must include:

- tenant ID;
- agent ID and display name;
- provider-neutral runtime (`codex`, `claude_code`, `opencode`, or `pi`);
- supported role;
- bounded biometric scope;
- accountable owner;
- documented purpose;
- machine contribution disclosure;
- human approval evidence for privileged roles.

BIOP stores agent registrations as tenant-scoped records and uses them as composition metadata for generated applications. Runtime-specific clients, prompts, credentials, and orchestration belong behind the AICR adapter contract.

### 9. Bytewax Lifecycle Batches

Generated applications may validate lifecycle batches for consent, template, verification, liveness, match-review, privacy-review, retention, and biometric-agent changes. BIOP accepts only Bytewax-routed lifecycle batches, requires at least one mutation, rejects unsupported lifecycle operations, and records accepted or denied batch evidence.

## Rule Engine

The capability contract must include deterministic rules for:

- tenant context;
- explicit consent;
- encrypted template storage;
- authentication liveness;
- cross-border privacy review;
- low-confidence match review;
- independent match reviewer;
- independent privacy reviewer;
- active consent for biometric operations;
- active template for verification.
- biometric governance-agent runtime, role, scope, owner, purpose, contribution disclosure, and privileged-role review;
- non-empty Bytewax lifecycle batches.

Rules are declarative guardrails for generated applications and publish-plan evidence. The runtime must enforce equivalent behavior.

## UI and Theming

BIOP must expose UI routes and theme components for:

- dashboard;
- users;
- consent center;
- enrollments;
- template vault;
- verification workbench;
- liveness workbench;
- match review queue;
- privacy review queue;
- biometric governance-agent roster;
- lifecycle batch monitor;
- compliance;
- analytics;
- settings.

Theme components must support compact enterprise controls, including modality status, consent scope, encrypted template vault, liveness score, match confidence, review queue, privacy posture, biometric governance-agent roster, and Bytewax lifecycle posture.

## Adapter Boundaries

The package runtime must not implement or require:

- real biometric capture hardware;
- raw face/voice/fingerprint/iris processing;
- model inference;
- external AI-agent runtime clients;
- production database persistence;
- HSM template encryption;
- legal/regulatory advice;
- production web server wiring.

Those concerns belong behind adapters. The package runtime defines the executable contract and governance semantics that adapters must honor.

## Acceptance Criteria

- The package has `SPECIFICATION.md` and `PLAN.md`.
- The package exposes dependency-light runtime, API-helper, and view-model surfaces.
- The contract, app semantic model, release report, and manifest include consent/review/template lifecycle surfaces.
- The contract, app semantic model, release report, and manifest include first-class biometric governance agents and Bytewax lifecycle batch metadata.
- Focused tests prove positive and negative governance paths.
- The package imports and self-tests without Flask, Flask-AppBuilder, SQLAlchemy session setup, biometric engines, or external hardware.
- Implementation audit and publish-plan pass for `capabilities/common/biop`.
