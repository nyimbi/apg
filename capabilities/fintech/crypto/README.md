# Cryptocurrency Services

`fintech_crypto` is the APG package-backed Cryptocurrency Services capability.
It provides executable digital-asset registry, custody account, balance, order,
trade, transfer, compliance screening, market price, review, and AI-agent
workflows for generated APG fintech applications.

The package is dependency-light and provider-neutral. It exposes a Python
contract, deterministic rules, runtime service methods, process-local API
helpers, UI view models, theme metadata, Bytewax lifecycle metadata, tests, and
release evidence without requiring live exchanges, custody providers, trading
venues, chain signing, market-data vendors, or blockchain RPC clients.

## What It Provides

- Crypto asset registration by symbol, type, network reference, token contract,
  precision, owner, and evidence.
- Custody account registration with custody model, provider reference, policy,
  owner, and evidence.
- Balance snapshots with fiat valuation and evidence.
- Order intake with side/type/quantity/limit price, requester, policy, and
  evidence.
- Trade execution records with venue, price, quantity, fees, status, and
  settlement reference.
- Transfer requests with approval, destination, status, and evidence.
- Compliance screening for wallets, transactions, assets, counterparties,
  sanctions, and travel-rule workflows.
- Price snapshots from exchange, oracle, custodian, manual, or aggregator
  sources.
- Provider-neutral crypto agents for Codex, Claude Code, OpenCode, and Pi.

## Local Usage

Inspect the APG contract:

```bash
./.venv/bin/apg capabilities inspect fintech_crypto --json
```

Run the local self-test:

```bash
./.venv/bin/python capabilities/fintech/crypto/app.py
```

Run focused tests:

```bash
./.venv/bin/pytest -q capabilities/fintech/crypto/tests/test_package_contract.py
```

Use the service directly:

```python
from capabilities.fintech.crypto import CryptocurrencyServicesService

service = CryptocurrencyServicesService()
asset = service.register_asset("asset-1", "tenant-a", "USDC", "stablecoin", "fintech_blockchain:polygon", "contract-ref", 6, "owner-a", "asset-evidence")
account = service.open_custody_account("account-1", "tenant-a", "custodian-ref", "mpc", "policy-ref", "owner-a", "custody-evidence")
order = service.create_order("order-1", "tenant-a", account["id"], asset["id"], "buy", "limit", 1000000, 100, "order-policy", "requester-a", "order-evidence")
service.record_trade("trade-1", "tenant-a", order["id"], "venue-ref", 100, 1000000, 10, "executed", "settlement-ref")
```

## Rule Engine

The deterministic rule engine is defined in `capability_contract.py` and
enforced by `service.py`. Rules cover tenant context, write-policy evidence,
asset symbol/type/network/precision/owner/evidence, custody provider/model/
policy/owner/evidence, balances, orders, limit prices, trades, transfers,
screening, prices, reviews, Bytewax batch routing, supported AI-agent runtimes
and roles, and human approval for privileged agent actions.

## Composition

The capability depends on APG auth, audit, notifications, NLP, keys,
Blockchain Services, wallets, risk, compliance, RegTech, AML, and KYC
contracts. Live exchange connectivity, custody-provider APIs, order routing,
signing, blockchain RPC, chain indexing, and market-data feeds remain adapter
responsibilities.
