# CKM Notification System Packet Plan

## Scope

Build the `ckm_not` capability as a coherent lifecycle and guardrail packet for
APG applications that need notification templates, campaigns, deliveries,
preferences, provider registration, analytics, AI-agent review, UI metadata,
theme metadata, Bytewax stream governance, and publishable package evidence.

## Implementation Packets

1. Specification and contract
   - Replace stale narrative in `cap_spec.md` with a pointer to the active
     specification.
   - Define the normative capability behavior in `SPECIFICATION.md`.
   - Expand `capability_contract.py` with configuration, rules, UI routes,
     theme metadata, provides/requires, and Bytewax streaming.

2. Dependency-light lifecycle
   - Add template, preference, delivery, and notification-agent data contracts.
   - Implement `NotificationLifecycleService` for template creation/approval,
     preference recording, delivery request classification, agent registration,
     batch mutation validation, audit events, and dashboard summary.
   - Keep live provider SDKs and stream workers behind adapters.

3. Package import hygiene
   - Fix the CKM namespace so the `not` directory is importable despite the
     Python keyword.
   - Keep the `ckm_not` package entrypoint dependency-light for contract and
     lifecycle usage.

4. Documentation and generated evidence
   - Add `README.md` usage and composition guidance.
   - Refresh app, semantic model, manifest, and release evidence from the live
     contract.
   - Update the progress log with proof commands and review notes.

5. Focused proof and review
   - Add a root package test that avoids legacy provider-heavy tests.
   - Run compile checks, focused tests, semantic probes, implementation audit,
     publish plan, stale-marker scan, and diff checks.
   - Review changed files for tenant context, consent enforcement, AI-agent
     boundaries, Bytewax guardrail coverage, and generated evidence consistency.

## Out Of Scope

- Live provider dispatch through email, SMS, push, voice, chat, webhook, or
  web-push networks.
- Durable database migrations for legacy SQLAlchemy models.
- Browser-rendered UI.
- Production Bytewax topology deployment.
- Full repository test suite.

## Review Checklist

- Contract is registry-valid and APG Python route metadata uses practical
  targets.
- `not` keyword import issue is resolved without importing heavy child modules
  at CKM namespace import time.
- All AI-agent guardrails include runtime, role, scope, registration, and
  contribution disclosure.
- Delivery rules honor tenant context, consent, preferences, quiet hours,
  provider secret references, and audit evidence.
- Batch mutations are rejected unless the event stream is Bytewax.
- Generated semantic evidence matches the executable contract.
- Documentation explains how to compose and extend the capability without
  binding live providers inside the package.
