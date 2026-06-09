# Blockchain Services

**Capability ID**: `fintech_blockchain` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Blockchain Services provides governed, multi-network blockchain infrastructure for fintech applications: network registration, wallet and custody management, smart contract deployment, on-chain transaction recording, evidence anchoring, oracle feed management, node health monitoring, and review workflows. It is deliberately provider-neutral — live chain RPC calls, signing keys, custody providers, and oracle connectivity remain adapter boundaries.

## Installation

```bash
pip install apg-fintech-blockchain
```

## Provides

- `blockchain_network_workflow`
- `blockchain_wallet_workflow`
- `smart_contract_workflow`
- `chain_transaction_workflow`
- `evidence_anchor_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-blockchain/dashboard` | `fintech_blockchain:view` | Overview |
| `/fintech-blockchain/networks` | `fintech_blockchain:networks` | Networks |
| `/fintech-blockchain/wallets` | `fintech_blockchain:wallets` | Custody |
| `/fintech-blockchain/contracts` | `fintech_blockchain:contracts` | Contracts |
| `/fintech-blockchain/transactions` | `fintech_blockchain:transactions` | Ledger |
| `/fintech-blockchain/anchors` | `fintech_blockchain:anchors` | Evidence |
| `/fintech-blockchain/oracles` | `fintech_blockchain:oracles` | Data |
| `/fintech-blockchain/nodes` | `fintech_blockchain:nodes` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_private_blockchain()`
- `deploy_smart_contract()`
- `invoke_smart_contract()`
- `record_transaction()`
- `verify_transaction()`
- `get_block()`
- `audit_trail_on_chain()`
- `verify_anchor()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_blockchain` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_blockchain;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_BLOCKCHAIN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
