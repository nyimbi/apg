# CONS Capability Specification

## Purpose

CONS defines the APG consent and privacy-management capability. It gives
generated applications an executable, governed lifecycle for privacy notices,
lawful purposes, consent capture, withdrawal, preferences, privacy requests,
consent-gated processing, AI privacy agents, and audit evidence.

## Scope

In scope:

- Tenant-aware in-memory service state for package checks and generated apps.
- Notice, purpose, consent, preference, privacy request, processing decision,
  privacy-agent, and audit models.
- Deterministic provenance hashes, consent-age checks, SLA due dates, request
  SLA posture, and consent coverage metrics.
- Deterministic rule evaluation for privacy guardrails.
- First-class AI privacy-agent registration with runtime, role, scope,
  disclosure, and policy reference.
- Bytewax stream contract metadata for lifecycle and batch mutation events.
- Framework-neutral API helpers and UI view models.
- Theme tokens and component metadata for generated APG applications.
- Package evidence through `app.py`, `semantic_model.json`,
  `package_manifest.json`, and `release_report.json`.

Out of scope for this dependency-light package:

- Live identity-provider checks.
- Production DLP scans.
- External audit-log sink writes.
- Notification or workflow dispatch.
- Regulator or marketing-platform integrations.
- Persistent database storage.
- Rendered browser UI.

Those behaviors attach through explicit adapters so local APG tooling stays
safe, deterministic, and side-effect free.

## Functional Requirements

### Notices

- Publish tenant-scoped privacy notices.
- Store version, URL, language, purposes, publishing actor, and timestamp.
- Support duplicate notice IDs across tenants.
- Emit audit evidence.

### Purposes

- Create tenant-scoped privacy purposes.
- Require owner, legal basis, retention policy, notice linkage, and data
  categories.
- Support active/inactive state changes only with reason and audit evidence.
- Support duplicate purpose IDs across tenants.

### Consents And Preferences

- Capture consent only for tenant-local purposes and notices.
- Require notice evidence.
- Store provenance hash.
- Withdraw consent with audit evidence.
- Update subject preferences for channels and purposes.

### Consent-Gated Processing

- Allow processing only when an active consent exists for tenant, subject, and
  purpose.
- Deny processing without active consent and record the decision.

### Privacy Requests

- Submit privacy requests with request type, subject, submitter, identity
  verification, evidence reference, and SLA due date.
- Deny requests without identity verification or evidence reference.
- Complete requests with resolution and audit evidence.

### AI Privacy Agents

- Register AI privacy agents as first-class CONS records.
- Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
- Supported roles: `notice_reviewer`, `consent_operator`,
  `privacy_request_reviewer`, `dlp_reviewer`, `compliance_reviewer`.
- Require registration flag, supported runtime, supported role, explicit scope,
  and contribution disclosure.
- Isolate agent registrations by tenant even when agent IDs are reused.

### UI And Theme

CONS must expose route metadata for:

- `dashboard`
- `purposes`
- `notices`
- `consents`
- `requests`
- `preferences`
- `agents`
- `analytics`
- `audit`
- `settings`

CONS must expose view-model functions for dashboard, subject privacy, privacy
agents, audit trail, analytics, and settings, and publish the
`cons_privacy_center` theme with purpose, consent, request, preference, agent,
and audit component metadata.

### Streaming

CONS must declare Bytewax as the lifecycle stream processor. The stream
contract must include purpose, notice, consent, preference, privacy request,
processing decision, privacy agent, and audit state families. Batch privacy
mutation must be denied unless the event stream is `bytewax`.

## Rule Engine Requirements

The deterministic rules must cover:

- tenant context;
- purpose legal basis, owner, retention policy, and notice linkage;
- consent capture notice evidence;
- active consent for consent-gated processing;
- identity verification and evidence for privacy requests;
- stale-consent review;
- AI privacy-agent registration, runtime, role, scope, and disclosure;
- state-change reason and audit evidence;
- cross-tenant access denial;
- Bytewax event stream requirement for batch mutation.

The rule evaluator must support equality plus numeric `_lt`, `_lte`, `_gt`,
`_gte`, and inequality `_ne` conditions.

## Non-Functional Requirements

- Importing the package must not require live adapters.
- Service operations must remain tenant-scoped.
- Generated package evidence must stay synchronized with the contract.
- API and view-model functions must return plain Python dictionaries/lists.
- Focused tests must cover the main lifecycle, guardrail failures, AI agents,
  tenant-safe duplicate IDs, Bytewax metadata, and generated evidence.
- Documentation must explain use, architecture, boundaries, and verification.

## Acceptance Criteria

- `README.md`, `SPECIFICATION.md`, `PLAN.md`, and `cap_spec.md` describe the
  same executable packet.
- `register_capability()` exposes dependencies, optional adapters,
  permissions, endpoints, UI metadata, theme, and Bytewax stream contract.
- Focused CONS tests pass.
- `app.self_test()` passes.
- `semantic_model.json` exposes CONS routes, rules, configuration, theme, and
  Bytewax stream metadata.
- Implementation audit and publish-plan pass for CONS.
- Stale-marker search finds no unsupported overclaims, unfinished markers, or
  unsupported stream-provider references in CONS.

