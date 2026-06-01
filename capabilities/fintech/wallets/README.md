# Digital Wallets

Digital Wallets provides APG-generated applications with tenant-scoped wallet
accounts, stored-value ledger entries, wallet instruments, transfers, holds,
limits, and governed AI wallet agents.

The package is dependency-light at the generated-application boundary. It can
run without live banks, payment processors, databases, web frameworks, or queue
brokers. Live integrations remain adapter work behind `walt`,
`fintech_payments`, `fintech_gateway`, `keym`, `auth`, `audl`, and `ntfy`.

## Runtime Files

- `capability_contract.py`: configuration, deterministic rules, routes, theme,
  dependencies, and Bytewax streaming metadata.
- `models.py`: wallet, instrument, ledger, and evidence dataclasses.
- `wallets_runtime.py`: Decimal and limit helper functions.
- `service.py`: executable wallet lifecycle and guardrail enforcement.
- `api.py`: process-local helper functions for generated applications.
- `views.py`: framework-neutral view models.
- `app.py`: semantic model, component manifest, and self-test.
- `tests/test_package_contract.py`: focused package tests.

## Public Lifecycle

1. `open_wallet`
2. `register_instrument`
3. `credit_wallet`
4. `debit_wallet`
5. `transfer`
6. `place_hold`
7. `release_hold`
8. `register_wallet_agent`
9. `validate_batch`
10. `dashboard_summary`

## Guardrails

The rule engine denies or requires review for missing tenant context, missing
write policy, missing wallet owners, unsupported wallet types, unsupported
currencies, missing wallet references, unsupported instruments, missing token
references, unverified instruments, non-positive credit/debit/hold/release
amounts, insufficient available balances, same-wallet transfers, cross-currency
transfers that need an FX workflow, large transfers without review, holds above
available balance, releases above held balance, non-Bytewax lifecycle batches,
unsupported wallet-agent runtimes or roles, and privileged agent actions without
human approval.

## AI Wallet Agents

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles include wallet operations, risk, limits, settlement, and dispute
reviewers. Agents can prepare and recommend actions, but privileged actions
require human approval evidence.

## Bytewax Streaming

All batch and lifecycle metadata uses:

- processor: `bytewax`
- stream: `apg.fintech.wallets.lifecycle`
- key: `tenant_id`

Kafka is intentionally not part of this package boundary.

## Example

```python
from capabilities.fintech.wallets import DigitalWalletsService

svc = DigitalWalletsService()
source = svc.open_wallet("wallet-a", "tenant-a", "customer-a", "consumer", "KES", 1000)
target = svc.open_wallet("wallet-b", "tenant-a", "merchant-b", "merchant", "KES", 0)
svc.transfer("transfer-1", "tenant-a", source["id"], target["id"], 125)
```
