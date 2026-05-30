# WALT Development Plan

## Goal

Make WALT a complete APG capability packet that generated applications can
compose for wallet ledgers, payment instruments, transaction authorization,
risk review, capture, settlement, reconciliation, AI-assisted review, Bytewax
events, UI surfaces, visual theming, documentation, and focused verification.

## Work Items

1. Documentation packet
   - Add `SPECIFICATION.md`.
   - Add `PLAN.md`.
   - Add `README.md`.
   - Replace `cap_spec.md` with the active lifecycle packet summary.

2. Contract expansion
   - Add `walt_agents`, `observability`, and `adapters` configuration.
   - Add provides/requires metadata.
   - Add Bytewax streaming manifest and event-stream helper.
   - Add deterministic rules for ledger evidence, compliance policy,
     instrument tokenization, instrument verification, transaction risk,
     transaction Bytewax streams, settlement approval, settlement Bytewax
     streams, reconciliation evidence, batch settlement, agents, and
     privileged agent actions.
   - Add `/walt/agents` and `/walt/policy` UI routes.

3. Runtime expansion
   - Add WALT agent records and metadata-rich audit events.
   - Extend `WaltService` with agent registration, privileged agent-action
     validation, batch settlement validation, Bytewax lifecycle metadata, and
     stronger wallet/instrument/transaction/settlement/reconciliation
     guardrails.
   - Keep production integration behind adapters.

4. API and views
   - Expose agent and batch validation helpers.
   - Add agent workbench and policy center view models.
   - Include streaming metadata and policy guardrails in dashboard,
     transaction, settlement, risk, settings, and status surfaces.

5. Generated evidence
   - Refresh `app.py`, `semantic_model.json`, `package_manifest.json`, and
     `release_report.json` from the expanded contract.
   - Ensure package manifest lists docs, contract, runtime, API, views, and
     tests.

6. Verification
   - Run focused py_compile for WALT package files.
   - Run focused WALT tests.
   - Run implementation audit for `capabilities/common/walt`.
   - Run publish-plan for `capabilities/common/walt`.
   - Run stale-marker and unsupported stream scans on touched WALT files.

## Review Checklist

- Tenant context is enforced.
- Wallet owner, ledger, and compliance references are enforced.
- Instruments require encryption, token references, and verifier attribution.
- Transactions require MFA for high value, risk evidence, and Bytewax routing.
- High-risk transactions route for review.
- Settlement requires captured transactions, reconciliation, approval, and
  Bytewax routing.
- Reconciliation has evidence references.
- WALT agents are first-class and constrained by runtime, role, and human
  approval policy.
- Generated app evidence matches the contract.
