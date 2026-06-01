# Robo Advisory

Robo Advisory is an executable APG capability for algorithm-guided investment
advice under suitability, policy, approval, audit, and AI-agent supervision. It
builds on Wealth Management by making model portfolios, recommendation packets,
automated rebalancing, drift monitoring, tax-loss harvesting, and review
evidence first-class application components.

The package is dependency-light. Generated applications can import the Python
service directly, while production deployments bind adapters for identity,
audit, notifications, language, keys, Wealth Management, KYC, AML, fraud,
analytics, reporting, market data, and Bytewax.

## Use

```python
from capabilities.fintech.robo import RoboAdvisoryService

service = RoboAdvisoryService()
profile = service.create_investor_profile(
    "profile-1", "tenant-1", "client-1", "kyc-1", "suitability-1", "balanced"
)
goal = service.define_goal_plan(
    "goal-1", "tenant-1", profile["id"], "retirement", 50000000, "USD",
    "2036-12-31"
)
model = service.publish_model_portfolio(
    "model-1", "tenant-1", "Balanced Core", "balanced",
    {"equity": 60, "fixed_income": 35, "cash": 5}, "policy-1"
)
recommendation = service.generate_recommendation(
    "rec-1", "tenant-1", profile["id"], goal["id"], model["id"], "analysis-1"
)
```

## Capability Surfaces

- Investor profile setup with KYC, suitability, and risk evidence.
- Goal planning with target amount, currency, and horizon.
- Model portfolio publication with allocation and policy controls.
- Recommendation packet generation and approval workflow.
- Auto-invest plan configuration.
- Drift monitoring and rebalance advice.
- Tax-loss harvesting candidates.
- Human review records and provider-neutral AI-agent registration.
- Deterministic rule engine and Bytewax lifecycle stream metadata.

## Boundaries

Live brokerage execution, market-data feeds, tax-lot accounting, custody,
statement generation, billing, regulator filing, and durable Bytewax workers are
adapter responsibilities, not direct package side effects.
