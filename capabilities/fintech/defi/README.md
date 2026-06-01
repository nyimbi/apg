# APG Decentralized Finance

`fintech_defi` is the APG package-backed capability for building governed
decentralized-finance applications. It composes protocol registries, DeFi
positions, governed actions, yield strategies, reward accruals, governance
votes, risk assessments, reviews, Bytewax lifecycle metadata, UI/view models,
visual theming, and provider-neutral AI-agent automation.

## What It Provides

- Protocol registry for lending pools, liquidity pools, staking, yield vaults,
  DEXs, bridges, derivatives, and insurance pools.
- Position ledger for supply, borrow, liquidity, stake, vault-share, long,
  short, and cover records.
- Action workflow for deposit, withdraw, borrow, repay, swap, stake, unstake,
  claim, and rebalance requests.
- Yield strategy, reward accrual, governance vote, risk assessment, and review
  workflows.
- Deterministic guardrails enforced before service state changes.
- UI route/view metadata and compact theme tokens for generated apps.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Bytewax lifecycle metadata through `apg.fintech.defi.lifecycle`.

## Use The Service

```python
from capabilities.fintech.defi import DecentralizedFinanceService

service = DecentralizedFinanceService()
protocol = service.register_protocol(
	"protocol-1",
	"tenant-a",
	"lending_pool",
	"fintech_blockchain:polygon",
	"aave-v3",
	"owner-1",
	"evidence://protocol",
	"medium",
)
position = service.open_position(
	"position-1",
	"tenant-a",
	protocol["id"],
	"wallet://treasury",
	"USDC/ETH",
	"supply",
	1_000_000,
	0,
	15_000,
	"evidence://position",
)
```

Invalid operations raise `PermissionError` with rule reasons such as
`tenant_context_required`, `protocol_type_not_supported`,
`action_approval_required`, or `bytewax_event_stream_required`.

## Compose In Generated Apps

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- View models: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

Generated applications should import the service or API helpers and render UI
from the route, screen, and theme metadata in the contract or semantic model.

## Verify Locally

```bash
./.venv/bin/pytest -q capabilities/fintech/defi/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/defi/app.py
./.venv/bin/apg capabilities inspect fintech_defi --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/defi --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/defi --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/defi --json
```

## Production Boundaries

Live protocol RPC, transaction signing, private-key custody, oracle data feeds,
on-chain liquidation execution, bridge execution, governance submission, MEV
protection, and durable Bytewax topology execution are adapter responsibilities.
Keep those integrations behind explicit APG contracts instead of embedding them
in this dependency-light package.
