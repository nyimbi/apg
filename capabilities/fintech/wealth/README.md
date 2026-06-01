# Wealth Management

Wealth Management is an executable APG capability for regulated advisory,
portfolio, mandate, order, performance, and fee workflows. It lets generated
applications compose client onboarding, suitability, investment-policy checks,
portfolio construction, rebalance review, order staging, performance reporting,
and advisor/AI supervision.

The package is dependency-light and can run inside generated Python
applications. Production deployments bind the adapter keys in the capability
contract to APG identity, audit, notifications, market data, custody, KYC, AML,
fraud, and Bytewax services.

## Use

```python
from capabilities.fintech.wealth import WealthManagementService

service = WealthManagementService()
client = service.register_client_profile(
    "client-1", "tenant-1", "Amina Client", "kyc-1", "tax-1", "risk-1"
)
suitability = service.capture_suitability_profile(
    "suitability-1", "tenant-1", client["id"], "balanced", "medium",
    "five_years", ["capital_growth", "income"]
)
portfolio = service.create_portfolio(
    "portfolio-1", "tenant-1", client["id"], "Core Portfolio", "USD",
    "advisor-1", "ips-1"
)
mandate = service.create_advisory_mandate(
    "mandate-1", "tenant-1", portfolio["id"], suitability["id"],
    "discretionary", "policy-1"
)
service.propose_rebalance(
    "rebalance-1", "tenant-1", portfolio["id"], mandate["id"],
    {"equity": 60, "fixed_income": 35, "cash": 5}, "analysis-1"
)
```

## Capability Surfaces

- Client profile onboarding with KYC, tax, and risk evidence.
- Suitability profile capture across objective, tolerance, horizon, and goals.
- Portfolio creation with currency, advisor, and investment-policy statement.
- Advisory mandate setup for advisory, discretionary, model, and execution-only
  arrangements.
- Allocation and rebalance workflow with drift and analysis evidence.
- Trade order staging with approval controls.
- Performance snapshot and fee schedule recording.
- Dashboard, portfolio console, advisor workbench, settings, and agent views.
- Deterministic rule engine and Bytewax lifecycle stream metadata.

## Integration Boundaries

Live custody, broker routing, market data, account aggregation, tax lots,
statement generation, billing collection, suitability questionnaires, and
durable Bytewax workers stay behind adapter boundaries.
