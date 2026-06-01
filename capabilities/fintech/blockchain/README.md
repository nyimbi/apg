# Blockchain Services

`fintech_blockchain` is the APG package-backed Blockchain Services capability.
It provides executable network registry, custody policy, smart contract,
transaction, evidence anchoring, oracle, node-health, review, and AI-agent
workflows for generated APG fintech applications.

The package is dependency-light and provider-neutral. It exposes a Python
contract, deterministic rules, runtime service methods, process-local API
helpers, UI view models, theme metadata, Bytewax lifecycle metadata, tests, and
release evidence without requiring live chain RPC calls, signing keys, custody
providers, chain indexers, bridge operators, or oracle vendors.

## What It Provides

- Blockchain network registration by type, environment, chain ID, endpoint
  reference, owner, and evidence.
- Wallet/custody registry with explicit custody model and key policy evidence.
- Smart contract deployment records with artifact, approval, and evidence.
- Chain transaction records with status, signer, asset, amount, and high-value
  approval guardrails.
- Evidence anchoring for payload hashes and APG reference IDs.
- Oracle feed registration and node-health recording.
- Provider-neutral blockchain agents for Codex, Claude Code, OpenCode, and Pi.
- UI route metadata and theme tokens for generated blockchain operations
  consoles.

## Local Usage

Inspect the APG contract:

```bash
./.venv/bin/apg capabilities inspect fintech_blockchain --json
```

Run the local self-test:

```bash
./.venv/bin/python capabilities/fintech/blockchain/app.py
```

Run focused tests:

```bash
./.venv/bin/pytest -q capabilities/fintech/blockchain/tests/test_package_contract.py
```

Use the service directly:

```python
from capabilities.fintech.blockchain import BlockchainServicesService

service = BlockchainServicesService()
network = service.register_network("net-1", "tenant-a", "ethereum", "testnet", "11155111", "rpc-ref", "owner-a", "network-evidence")
wallet = service.register_wallet("wallet-1", "tenant-a", network["id"], "wallet-ref", "mpc", "key-policy", "owner-a", "wallet-evidence")
contract = service.deploy_contract("contract-1", "tenant-a", network["id"], "settlement", "artifact-ref", "owner-a", "approval-ref", "contract-evidence")
service.record_transaction("tx-1", "tenant-a", network["id"], "0xabc", "settlement", contract["id"], 1000, "signer-a", "tx-evidence", "confirmed")
```

## Rule Engine

The deterministic rule engine is defined in `capability_contract.py` and
enforced by `service.py`. Rules cover tenant context, write-policy evidence,
network type/environment/chain ID/RPC/owner/evidence, wallet custody and key
policy, smart contract artifact/approval/evidence, transaction hash/type/asset/
amount/signer/status/evidence/high-value approval, evidence anchors, oracle
feeds, node health, reviews, Bytewax batch routing, supported AI-agent runtimes
and roles, and human approval for privileged agent actions.

## Composition

The capability depends on APG auth, audit, notifications, NLP, keys, risk,
compliance, RegTech, and wallets contracts. Live chain RPC access, key custody,
transaction signing, chain indexing, bridge operation, and oracle connectivity
remain adapter responsibilities.
