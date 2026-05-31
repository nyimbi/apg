# Digital Forms and eSign Capability Specification

## Purpose

`esgn` is the APG common capability for governed digital forms and electronic signatures. It lets generated applications compose tenant-scoped form templates, submissions, signing envelopes, ordered signing ceremonies, first-class provider-neutral signing agents, evidence packages, audit events, UI screens, visual theming, and Bytewax lifecycle policy.

## Scope

The capability must support:

- Tenant-local form templates with accountable owner, readable name, schema fields, compliance framework, DLP policy, retention policy, review state, and publication state.
- Form submissions that validate against published template schemas and carry audit evidence references.
- Signature envelopes that bind a submission, document hash, expiry, sender, subject, recipients, recipient consent, delegated policy references, routing order, compliance review state, and tamper seal.
- Signing ceremonies that require verified identity, explicit signature intent, active/non-expired envelope state, valid tamper seal, and routing-order readiness.
- Envelope cancellation and rejection with mandatory reason and audit event.
- Evidence-package creation only after all required signatures complete, with encryption, retention, audit trail, seal digest, and completion certificate ID.
- First-class signing agents for form assistance, clause review, routing coordination, evidence audit, compliance review, signer-experience review, lifecycle-batch review, and signing stewardship.
- Provider-neutral signing-agent runtimes `codex`, `claude_code`, `opencode`, and `pi` through an AICR adapter contract, without invoking external agent clients from the package runtime.
- Privileged signing-agent roles that require human approval evidence and remain in `pending_review` without it.
- Bytewax-backed lifecycle stream configuration for template, submission, envelope, signing, evidence, signing-agent, and audit batches.
- UI route contracts and dependency-light view models for generated applications.

## Dependencies

Required:

- `auth` for identity, signer, and permission composition.
- `encr` for evidence encryption and seal storage composition.
- `audl` for audit sink composition.
- `comp` for compliance framework and retention composition.

Optional:

- `wflo`, `ntfy`, `idfd`, `dlpd`, `nlpc`, and `them`.

## Configuration

The authoritative configuration lives in `capability_contract.py` and includes:

- `forms`
- `submissions`
- `envelopes`
- `signatures`
- `evidence`
- `signing_agents`
- `agents`
- `streaming`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

## Rules

The deterministic rule engine covers:

- tenant context
- template owner, name, schema, compliance framework, DLP, approval, and regulated-form review
- submission schema and audit evidence
- envelope subject, recipient count, document hash, expiry, recipient consent, delegated signing policy, and state-change audit
- signer identity, signature intent, active envelope state, routing order, duplicate-recipient signatures, tamper seal, and expiry
- evidence encryption, completed envelope state, valid seal, audit trail, and retention policy
- cancellation/rejection reason capture
- first-class signing-agent ID, name, runtime support, role support, scope, owner, purpose, disclosure, and privileged-role approval
- ESGN lifecycle-batch mutation, operation, and Bytewax stream enforcement
- cross-tenant access denial
- legacy Bytewax batch mutation enforcement

## Runtime

`service.EsgnService` is the generated-application runtime. It stores deterministic in-memory state for:

- form templates
- submissions
- signature envelopes
- signing ceremonies
- evidence packages
- signing agents
- lifecycle batches
- audit events

The runtime enforces the same guardrails exposed by the contract rule engine and keeps live providers behind adapter boundaries.

## UI

The UI contract exposes:

- dashboard
- forms
- builder
- submissions
- envelopes
- signing
- agents
- lifecycle
- evidence
- audit
- analytics
- settings

## Production Boundary

This packet does not perform live identity proofing, durable document storage, cryptographic key custody, payment-grade notarization, notification delivery, browser-rendered signing, external AI-agent CLI execution, or live Bytewax worker execution. Those are production adapters behind the APG composition layer.

## Acceptance Gates

- `README.md`, `SPECIFICATION.md`, and `PLAN.md` describe the package clearly.
- `capability_contract.py` exposes configuration, deterministic rules, UI, theme, streaming, and adapter metadata.
- Runtime/API/view tests prove positive lifecycle behavior and negative guardrail behavior.
- `semantic_model.json`, `package_manifest.json`, and `release_report.json` match the current contract.
- Focused compile, pytest, implementation audit, publish-plan, stale-marker scan, and diff check pass.
