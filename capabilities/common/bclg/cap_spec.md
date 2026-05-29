# Blockchain Ledger Services Capability Specification

- **Capability Name**: Blockchain Ledger Services
- **Capability ID**: `bclg`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package implements the executable APG contract for `bclg` as a
dependency-light blockchain ledger runtime. It provides tenant ledger network
registration, managed key-custody binding, deterministic transaction hashing,
high-value transaction review, smart contract deployment governance, audit
events, UI route metadata, semantic-model publication, and publish-plan
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
key-custody binding.

## Rules

- `tenant_context_required`
- `ledger_requires_owner`
- `transaction_requires_signature`
- `key_custody_required`
- `contract_requires_review`
- `high_value_transaction_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model. The dashboard view model surfaces
ledger, key-custody, transaction, contract, audit, and review-queue state from
`BclgService`.

## Theme

The package uses the `bclg_ledger_ops` APG theme contract.

## Runtime Behavior

`BclgService` maintains deterministic in-memory registries for ledgers,
custody bindings, transactions, smart contract artifacts, ledger heads, and
audit events. Transactions receive stable SHA-256 hashes from `ledger_engine.py`;
committed transactions update a deterministic block hash, while high-value
transactions remain in `pending_review` until approved. Smart contract
deployment requires review evidence, an artifact hash, rollback plan, and an
active custody binding for the target ledger.

## Known Integration Boundary

This package intentionally avoids live blockchain, HSM, or custody-provider
network calls. External chain anchoring, hardware key custody, wallet
settlement, and regulatory reporting should be composed through APG
capabilities such as `keym`, `encr`, `walt`, `audl`, and `comp`.
