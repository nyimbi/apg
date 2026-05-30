# Vendor Management Implementation Plan

## Delivery Slice

Build one coherent lifecycle and guardrail packet for `scm_ven`:

1. Replace generated contract metadata with an explicit APG capability contract.
2. Replace adapter-heavy service/API/view entry points with dependency-light package surfaces.
3. Add missing `SPECIFICATION.md` and `PLAN.md`, and refresh README guidance.
4. Refresh semantic model, package manifest, and release evidence from the executable contract.
5. Add focused tests and run battery-conscious verification.

## Design Choices

- Use Python as the capability target.
- Keep database, procurement, contract, document, risk, web, and notification adapters outside the import path.
- Use deterministic guardrails for lifecycle correctness.
- Use Bytewax event metadata for all batches and lifecycle streams.
- Treat vendor review agents as lifecycle records with runtime, role, scope, and approval constraints.

## Acceptance Criteria

- Contract shape validates.
- Lifecycle service creates vendor, qualification, onboarding, performance, risk, compliance, contract, communication, portal user, scorecard, and agent records.
- Rules reject missing context, unsupported values, unapproved or incomplete records, non-Bytewax batch routing, and invalid agent actions.
- App self-test passes.
- APG publish plan and implementation audit pass for the package.

## Deferred Work

- Live supplier master, procurement, sourcing, contract, document, risk, audit, workflow, notification, and persistence adapters.
- Rendered UI/browser verification.
- Performance/load testing.
- Persistent database migrations.
