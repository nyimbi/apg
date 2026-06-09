# Decentralized Finance

**Capability ID**: `fintech_defi` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Decentralized Finance provides governed operations over DeFi protocols: protocol registry, position management (supply, borrow, liquidity, stake, vault share), action execution workflow (deposit, withdraw, borrow, repay, swap, stake, unstake, claim, rebalance), yield strategy management, reward accruals, governance voting, risk tier assessments, and reviews. Every action against a DeFi protocol requires an approval reference before it is recorded, enforcing human oversight over autonomous on-chain interactions.

## Installation

```bash
pip install apg-fintech-defi
```

## Provides

- `defi_protocol_workflow`
- `defi_position_workflow`
- `defi_action_workflow`
- `defi_yield_strategy_workflow`
- `defi_reward_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-defi/dashboard` | `fintech_defi:view` | Overview |
| `/fintech-defi/protocols` | `fintech_defi:protocols` | Protocols |
| `/fintech-defi/positions` | `fintech_defi:positions` | Portfolio |
| `/fintech-defi/actions` | `fintech_defi:actions` | Operations |
| `/fintech-defi/yield-strategies` | `fintech_defi:yield` | Strategies |
| `/fintech-defi/rewards` | `fintech_defi:rewards` | Portfolio |
| `/fintech-defi/governance` | `fintech_defi:governance` | Governance |
| `/fintech-defi/risk` | `fintech_defi:risk` | Risk |

## Key Service Methods

- `describe()`
- `evaluate()`
- `liquidity_pool_deposit()`
- `liquidity_pool_withdraw()`
- `yield_farming_enrol()`
- `claim_farming_rewards()`
- `lending_deposit()`
- `borrow_against_collateral()`
- `repay_loan()`
- `collateral_health_factor()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_defi` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_defi;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_DEFI_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
