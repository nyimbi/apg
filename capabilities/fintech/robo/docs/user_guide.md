# Robo Advisory

**Capability ID**: `fintech_robo` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Robo Advisory provides algorithm-guided investment advice under governance: investor profile creation with KYC and suitability evidence, goal planning, model portfolio publication with exact 100% allocation totals, recommendation generation and approval workflows, automated investment plan configuration, portfolio drift monitoring, tax-loss harvesting candidate recording, and governance reviews. It builds on Wealth Management by making model-driven recommendations, automated rebalancing, and tax optimization first-class governed operations.

## Installation

```bash
pip install apg-fintech-robo
```

## Provides

- `robo_investor_profile_workflow`
- `robo_goal_plan_workflow`
- `robo_model_portfolio_workflow`
- `robo_recommendation_workflow`
- `robo_automation_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-robo/dashboard` | `fintech_robo:view` | Overview |
| `/fintech-robo/profiles` | `fintech_robo:profiles` | Investors |
| `/fintech-robo/goals` | `fintech_robo:goals` | Investors |
| `/fintech-robo/models` | `fintech_robo:models` | Models |
| `/fintech-robo/recommendations` | `fintech_robo:recommendations` | Advice |
| `/fintech-robo/automation` | `fintech_robo:automation` | Advice |
| `/fintech-robo/drift` | `fintech_robo:drift` | Operations |
| `/fintech-robo/tax-loss` | `fintech_robo:tax_loss` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `risk_questionnaire()`
- `determine_risk_profile()`
- `recommended_portfolio()`
- `auto_invest()`
- `auto_rebalance()`
- `goal_tracking()`
- `onboard_client()`
- `drift_monitoring()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_robo` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_robo;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_ROBO_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
