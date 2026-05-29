# BCLG Capability Development Plan

## Current State

BCLG has a dependency-light ledger runtime with ledger registration,
key-custody binding, deterministic transaction/block hashes, high-value
transaction review state, smart contract deployment, contract/rule/theme
metadata, API helpers, view models, package evidence, and tests.

The package-level composition gap is that high-value transaction review and
smart contract deployment review can be represented by caller-supplied strings
or booleans instead of first-class approval state. Mutable stores are keyed by
raw IDs instead of tenant plus ID, so duplicate tenant-local IDs can collide.
Generated APG applications need a fail-closed, composable lifecycle for ledger
registration, custody, signed transactions, explicit reviews, contract
deployment, view models, semantic evidence, and audit state.

## Packet 1: Governed Ledger-Mutation Lifecycle

Deliver a focused lifecycle packet:

- add package-level high-value transaction review approval state;
- add package-level smart contract deployment approval state;
- key ledgers, custody bindings, transactions, contracts, approvals, ledger
  heads, and audit events by tenant plus record ID;
- prevent `transaction_review_recorded=True` from committing high-value
  transactions without explicit approved review evidence;
- approve or reject transaction review requests with independent reviewer notes;
- commit reviewed transactions only when approved;
- prevent duplicate pending transaction reviews and stale review decisions from
  mutating rejected or committed transactions;
- deploy smart contracts only with approved matching deployment approval
  evidence, artifact hash, rollback plan, and active key custody;
- update API helpers and view models for review queues and approval state;
- update contract routes, rules, theme metadata, semantic evidence, and release
  evidence;
- rename generated-package tests to package contract naming;
- update package documentation and progress evidence.

## Implementation Steps

1. Extend `models.py` with `TransactionReviewApproval` and
   `ContractDeploymentApproval`.
2. Update `service.py` so ledgers, custody bindings, transactions, transaction
   approvals, contract approvals, contracts, ledger heads, and audit events are
   tenant-qualified.
3. Add transaction review request/decision behavior and enforce review evidence
   before committing high-value transactions.
4. Add contract deployment review request/decision behavior and enforce review
   evidence before deployment.
5. Preserve deterministic transaction, contract, and block hash behavior.
6. Extend `api.py` for generated application calls.
7. Extend `views.py` for dashboard, ledger console, transaction monitor,
   transaction review queue, contract registry, contract approval queue, key
   custody, compliance, and audit surfaces.
8. Update `capability_contract.py` with review routes, independent-reviewer
   rules, and theme components.
9. Update registration metadata with review capabilities, endpoints, and
   permissions.
10. Replace stale embedded semantic evidence in `app.py` with contract-derived
    evidence.
11. Extend package tests with positive ledger-custody-standard-transaction,
    high-value-review, contract-approval-deployment, API-helper, view-model, and
    duplicate-ID tenant-isolation coverage.
12. Add negative missing-owner, missing-signature, missing-custody,
    caller-boolean review bypass, rejected transaction review, missing review
    notes, self-review, missing contract approval, rejected contract approval,
    tenant-mismatch, and duplicate same-tenant ID coverage.
13. Rename generated-package tests to package contract naming.
14. Update `cap_spec.md` with the current executable lifecycle and proof
    commands.
15. Run focused package proof, implementation audit, publish-plan, code review,
    fixes, and diff checks.

## Review Checklist

- Ledger, custody, transaction, approval, contract, head, and audit state is
  tenant-qualified.
- High-value transaction commit requires approved matching review evidence.
- Contract deployment requires approved matching deployment evidence.
- Reviewers cannot approve their own transaction or deployment requests.
- Review decisions require reviewer identity and notes.
- Rejected reviews cannot be converted into committed ledger changes.
- Duplicate pending transaction reviews are rejected.
- Caller-supplied booleans do not bypass review governance.
- Missing tenant context, owner, signature, custody, artifact hash, and rollback
  plan fail closed.
- API helpers expose the same behavior as service methods.
- View models expose ledger, custody, transaction, approval, contract,
  compliance, theme, and audit state.
- Live chain nodes, HSMs, wallets, compliance engines, and web servers remain
  adapter boundaries.
