# Robo Advisory — User Guide

**Capability ID**: `fintech_robo` | **Domain**: `fintech` | **Version**: `2.0.0`
**Author**: Nyimbi Odero | **© 2025 Datacraft** | www.datacraft.co.ke

---

## Description

Robo Advisory provides algorithm-guided investment advice under governance: investor profile creation with KYC and suitability evidence, goal planning, model portfolio publication, recommendation generation and approval workflows, automated investment plan configuration, portfolio drift monitoring, tax-loss harvesting, client analytics, and governance reviews.

Version 2.0.0 adds quantitative simulation (Monte Carlo, stress testing, sensitivity analysis), lifecycle glide paths, drawdown circuit breakers, robo-to-human escalation, wash-sale-safe tax-lot harvesting, income reinvestment, and client lifetime value / churn prediction.

---

## Installation

```bash
pip install apg-fintech-robo
```

---

## Quick Start

```python
import asyncio
from capabilities.fintech.robo.service import RoboAdvisoryService

svc = RoboAdvisoryService(tenant_id="my-tenant", actor_id="adviser-001")

async def main():
    # 1. Onboard a new client
    record = await svc.onboard_client(
        customer_id="cust-123",
        plan={
            "risk_profile": "balanced",
            "initial_investment_usd": 10_000,
            "monthly_contribution_usd": 500,
            "target_amount_usd": 100_000,
            "horizon_years": 10,
            "currency": "USD",
            "goal_type": "retirement",
            "cadence": "monthly",
        },
    )
    profile_id = record["profile_id"]
    goal_id = record["goal_id"]

    # 2. Run a Monte Carlo simulation
    mc = await svc.monte_carlo_retirement_simulation(profile_id, n_paths=5000)
    print(f"P50 projected value: ${mc['percentiles_usd']['p50']:,.0f}")
    print(f"Probability of success: {mc['probability_of_success']:.1%}")

    # 3. Stress test the portfolio
    stress = await svc.portfolio_stress_test(profile_id)
    print(f"Worst-case drawdown: {stress['worst_case']['drawdown_pct']:.1f}%")

asyncio.run(main())
```

---

## Risk Questionnaire and Profiling

```python
# Score a questionnaire
q = await svc.risk_questionnaire(
    customer_id="cust-456",
    responses={
        "investment_horizon": "5_to_10yr",
        "loss_reaction":      "hold",
        "income_stability":   "stable",
        "prior_experience":   "moderate",
        "savings_rate":       "medium",
    },
)
print(q["derived_risk_profile"])  # "balanced"

# Finalise the profile
profile_detail = await svc.determine_risk_profile(q["questionnaire_id"])
print(profile_detail["expected_annual_return_pct"])  # ~8.5
```

---

## Portfolio Recommendations

```python
rec = await svc.recommended_portfolio(
    risk_profile="aggressive",
    investment_amount=50_000,
    time_horizon_years=15,
    currency="KES",
)
# Returns holdings breakdown, projected value, fee schedule
```

---

## Auto-Invest and Auto-Rebalance

```python
# Schedule a monthly auto-investment
inv = await svc.auto_invest(
    customer_id="cust-123",
    amount=500.0,
    frequency="monthly",
    currency="USD",
)

# Rebalance a portfolio (dry-run first to preview trades)
preview = await svc.auto_rebalance(profile_id, dry_run=True)
for trade in preview["trades"]:
    print(f"{trade['direction'].upper()} {trade['asset_class']}: ${trade['trade_value_usd']:,.0f}")

# Execute rebalance
await svc.auto_rebalance(profile_id, drift_threshold_pct=5.0)
```

---

## Goal Tracking and Sensitivity Analysis

```python
# Track progress toward a goal
progress = await svc.goal_tracking(goal_id)
print(f"Progress: {progress['progress_pct']:.1f}%  On track: {progress['on_track']}")
print(f"Required monthly: ${progress['required_monthly_contribution_usd']:,.0f}")

# Sensitivity analysis: which lever matters most?
sensitivity = await svc.goal_sensitivity_analysis(goal_id)
print(f"Highest-leverage lever: {sensitivity['highest_leverage_lever']}")
print(f"Overall success rate across 27 scenarios: {sensitivity['overall_success_rate']:.1%}")
```

---

## Lifecycle Glide Path

```python
glide = await svc.compute_glide_path_allocation(
    profile_id=profile_id,
    goal_id=goal_id,
    current_age=45,
)
print(glide["glide_path_allocation"])
# {'equities': 39.0, 'government_bonds': 42.7, 'money_market': 12.2, 'cash': 6.1}
print(f"Rebalance required: {glide['rebalance_required']}")
```

---

## Monte Carlo Retirement Simulation

```python
mc = await svc.monte_carlo_retirement_simulation(
    profile_id=profile_id,
    n_paths=10_000,
    contribution_monthly_usd=500.0,
)
percentiles = mc["percentiles_usd"]
print(f"Bear case (P10):  ${percentiles['p10']:>12,.0f}")
print(f"Median   (P50):  ${percentiles['p50']:>12,.0f}")
print(f"Bull case (P90):  ${percentiles['p90']:>12,.0f}")
print(f"Probability of hitting goal: {mc['probability_of_success']:.1%}")
```

---

## Stress Testing

```python
stress = await svc.portfolio_stress_test(profile_id)
for result in stress["results"]:
    print(f"{result['scenario']}: -{result['drawdown_pct']:.1f}% drawdown, "
          f"~{result['estimated_recovery_months']:.0f}mo recovery")

# Custom scenario
custom = await svc.portfolio_stress_test(profile_id, scenarios=[{
    "name": "EAST_AFRICA_CORRECTION_2025",
    "description": "NSE correction: equities -30%, KES depreciation shock",
    "shocks": {"equities": -30.0, "government_bonds": -5.0, "money_market": 2.0, "cash": 0.5},
}])
```

---

## Drawdown Circuit Breaker

```python
# Check whether automation should be suspended
check = await svc.drawdown_circuit_breaker_check(
    portfolio_id=profile_id,
    peak_value_usd=15_000.0,
    drawdown_threshold_pct=15.0,
)
if check["circuit_open"]:
    print("Automation suspended:", check["recommendation"])
```

---

## Robo-to-Human Escalation

```python
escalation = await svc.evaluate_escalation_triggers(profile_id)
if escalation["escalate"]:
    print(f"Escalate to human adviser — triggers: {[t['trigger'] for t in escalation['triggers']]}")
    print(f"Automation frozen: {escalation['automation_frozen']}")
```

---

## Tax Optimisation

```python
# High-level TLH report
tax = await svc.tax_optimisation(profile_id, jurisdiction="KE")
print(f"Total harvestable loss: ${tax['total_harvestable_loss_usd']:,.0f}")
print(f"Estimated tax saving:   ${tax['total_estimated_tax_saving_usd']:,.0f}")

# Lot-level engine with wash-sale compliance
lots = await svc.tax_lot_harvesting_engine(
    profile_id=profile_id,
    jurisdiction="US",
    min_lot_age_days=31,
    min_loss_usd=500.0,
)
for candidate in lots["candidates"]:
    print(f"{candidate['asset_class']}: loss=${candidate['unrealised_loss_usd']:,.0f}, "
          f"saving=${candidate['estimated_tax_saving_usd']:,.0f}, "
          f"replace with {candidate['replacement_instrument']}")
```

---

## Income Reinvestment

```python
# Estimate and reinvest quarterly income
result = await svc.reinvest_income(portfolio_id=profile_id)
print(f"Total income credited: ${result['total_income_usd']:,.2f}")

# Or pass explicit dividend events
result = await svc.reinvest_income(
    portfolio_id=profile_id,
    income_events=[
        {"asset_class": "equities", "amount_usd": 320.0, "type": "dividend"},
        {"asset_class": "government_bonds", "amount_usd": 180.0, "type": "coupon"},
    ],
)
```

---

## Client Analytics

```python
# Lifetime value
clv = await svc.client_lifetime_value(customer_id="cust-123")
print(f"CLV: ${clv['lifetime_value_usd']:,.0f}  Annual fee revenue: ${clv['annual_fee_revenue_usd']:,.0f}")

# Churn prediction
churn = await svc.churn_probability(customer_id="cust-123")
print(f"Churn probability: {churn['churn_probability']:.1%} ({churn['churn_risk_tier']})")
print(f"Recommended intervention: {churn['recommended_intervention']}")
```

---

## Compliance and Reporting

```python
# Suitability check
suit = await svc.compliance_suitability_check(profile_id)
print(f"Suitable: {suit['suitable']}  Max deviation: {max(abs(v) for v in suit['deviations'].values()):.1f}pp")

# Performance report with alpha and Sharpe
perf = await svc.robo_performance_report(profile_id, period="2025-Q1")
print(f"Alpha: {perf['alpha_pct']:.2f}pp  Sharpe: {perf['sharpe_ratio']:.2f}")

# Fee transparency (MiFID II / CMA)
fees = await svc.fee_transparency_report(profile_id)
print(f"Annual management fee: ${fees['annual_management_fee_usd']:,.2f}")

# Portfolio health score
health = await svc.portfolio_health_score(profile_id)
print(f"Health score: {health['health_score']}/100  Components: {health['components']}")
```

---

## Drift Monitoring

```python
drift = await svc.drift_monitoring(profile_id, tolerance_pct=5.0)
print(f"Max drift: {drift['max_drift_pct']:.1f}%  Breaches: {drift['breach_count']}")
if drift["rebalance_recommended"]:
    for breach in drift["breaches"]:
        print(f"  {breach['asset_class']}: {breach['direction']} by {abs(breach['drift_pct']):.1f}pp")
```

---

## Dashboard Summary

```python
summary = svc.dashboard_summary(tenant_id="my-tenant")
print(summary)
# {
#   "profile_count": 42,
#   "goal_count": 38,
#   "auto_invest_executions": 156,
#   "rebalance_executions": 21,
#   "onboarded_clients": 42,
#   "tax_optimisations": 8,
#   ...
# }
```

---

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
| `/fintech-robo/stress-test` | `fintech_robo:analysis` | Analysis |
| `/fintech-robo/monte-carlo` | `fintech_robo:analysis` | Analysis |
| `/fintech-robo/analytics` | `fintech_robo:analytics` | Analytics |

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_ROBO_`.

| Key | Default | Description |
|-----|---------|-------------|
| `FINTECH_ROBO_DRIFT_THRESHOLD_BPS` | `500` | Drift threshold triggering rebalance recommendation |
| `FINTECH_ROBO_DRAWDOWN_CIRCUIT_PCT` | `15` | Drawdown % from peak that triggers circuit breaker |
| `FINTECH_ROBO_MC_N_PATHS` | `10000` | Monte Carlo path count |
| `FINTECH_ROBO_ESCALATION_AUM_USD` | `500000` | AUM threshold for human escalation |
| `FINTECH_ROBO_CGT_RATE` | `0.15` | Default CGT rate (Kenya) |
| `FINTECH_ROBO_MGMT_FEE_PCT` | `0.5` | Annual management fee % |
| `FINTECH_ROBO_HURDLE_RATE_PCT` | `6.0` | Performance fee hurdle rate % |

---

## Interoperability

`fintech_robo` integrates with other APG capabilities through the composition engine:

```apg
use fintech_robo;
```

Upstream dependencies: `fintech_wealth` (client context), `fintech_kyc` (identity verification), `fintech_aml` (AML screening).
Downstream consumers: `fintech_portfolio` (model allocation templates), `fintech_trading` (order execution from automation plans).

---

## Further Reading

- `service.py` — Business logic implementation (all async methods)
- `models.py` — Dataclass models (InvestorProfile, GoalPlan, ModelPortfolio, ...)
- `api.py` — REST API endpoints (Flask-AppBuilder blueprints)
- `views.py` — Pydantic v2 request/response schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 quantitative and operational improvements
- `README.md` — Capability contract, business rules, streaming events
