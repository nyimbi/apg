# Blockchain Ledger Services Capability Specification

- **Capability Name**: Blockchain Ledger Services
- **Capability ID**: `bclg`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package implements the executable APG contract for `bclg` as a
dependency-light blockchain ledger runtime. It provides tenant ledger network
registration, managed key-custody binding, deterministic transaction hashing,
high-value transaction review approval, smart contract deployment approval,
audit events, UI route metadata, semantic-model publication, and publish-plan
evidence without requiring an external blockchain node.

## Provided Services

- `ledger_registry`
- `transaction_signing`
- `smart_contract_governance`
- `key_custody`
- `ledger_audit`
- `capability_rules`

## Required Services

- `tenant_context`
- `encryption_policy`
- `key_management`
- `compliance_mapping`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations. Ledger creation requires an owner, consensus profile, and network
policy. Mutating transactions and contract deployment require an active
key-custody binding. High-value transaction commit and smart contract
deployment require explicit approval evidence with independent reviewer notes;
caller-supplied booleans or strings are not sufficient governance evidence.

## Rules

- `tenant_context_required`
- `ledger_requires_owner`
- `transaction_requires_signature`
- `key_custody_required`
- `contract_requires_review`
- `high_value_transaction_requires_review`
- `transaction_review_requires_independent_reviewer`
- `contract_deployment_review_requires_independent_reviewer`

## UI

The package exposes 10 APG Python UI route contract(s) through
`views.py` and the package semantic model. The dashboard view model surfaces
ledger, key-custody, transaction, transaction-review, contract,
contract-review, audit, and review-queue state from `BclgService`.

## Theme

The package uses the `bclg_ledger_ops` APG theme contract.

## Runtime Behavior

`BclgService` maintains deterministic tenant-qualified in-memory registries for
ledgers, custody bindings, transactions, transaction reviews, contract
deployment approvals, smart contract artifacts, ledger heads, and audit events.
Transactions receive stable SHA-256 hashes from `ledger_engine.py`; committed
transactions update a deterministic tenant-local block hash, while high-value
transactions remain in `pending_review` until approved by an independent
reviewer. Only one pending review can exist for a high-value transaction, and
rejected high-value transactions remain rejected and never update the ledger
head. Smart contract deployment requires approved matching deployment approval,
artifact hash, rollback plan, and active custody binding for the target ledger.

Current package-backed lifecycle:

1. Register tenant ledgers with owner, consensus profile, policy, participants,
   and fork-monitoring posture.
2. Bind active key custody to the tenant ledger before mutation.
3. Submit signed standard transactions and commit them with deterministic
   transaction and block hashes.
4. Submit high-value transactions into `pending_review` regardless of
   caller-supplied `transaction_review_recorded` booleans.
5. Request and decide high-value transaction reviews with independent reviewer
   notes.
6. Commit approved reviewed transactions and keep rejected transactions out of
   the ledger head; duplicate pending reviews and stale review decisions are
   rejected.
7. Request and decide smart contract deployment approvals with artifact hash,
   rollback plan, requester, reviewer, and notes.
8. Deploy contracts only when approved matching deployment evidence exists.
9. Keep ledger, custody, transaction, approval, contract, head, and audit state
   tenant-qualified so duplicate IDs across tenants cannot collide.
10. Emit tenant-scoped audit events for ledger, custody, transaction, review,
    contract, and compatibility lifecycle changes.

Focused proof commands:

```bash
./.venv/bin/pytest -q capabilities/common/bclg/test_capability_contract.py capabilities/common/bclg/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/bclg --json
./.venv/bin/apg capabilities publish-plan capabilities/common/bclg --json
git diff --check -- capabilities/common/bclg
```

## Known Integration Boundary

This package intentionally avoids live blockchain, HSM, or custody-provider
network calls. External chain anchoring, hardware key custody, wallet
settlement, and regulatory reporting should be composed through APG
capabilities such as `keym`, `encr`, `walt`, `audl`, and `comp`.
