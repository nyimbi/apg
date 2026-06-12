# Robo Advisory

## Overview

Algorithm-guided investment advice under governance: investor profile creation with KYC and suitability evidence, goal planning, model portfolio publication with exact 100% allocation totals, recommendation generation and approval workflows, automated investment plan configuration, portfolio drift monitoring, tax-loss harvesting, and governance reviews.

Recommendations require an approved model and analysis before generation. Automation plans require an approved recommendation. Model allocations must total exactly 100%. All robo advisory events stream to `apg.fintech.robo.lifecycle` via Bytewax.

## Capability ID

`fintech_robo`  Version: 2.0.0

## Features

- Risk questionnaire scoring across 5 dimensions with derived profile
- Lifecycle glide path engine: equity allocation shifts with age and goal horizon
- Monte Carlo retirement projection (10,000 paths, P10–P90 percentile fan)
- Portfolio stress testing: GFC 2008, COVID 2020, Rate Shock 2022
- Goal sensitivity analysis: 3D grid (return, contribution, horizon)
- Drawdown circuit breaker: auto-suspend on >15% drawdown from peak
- Tax-lot harvesting engine with wash-sale compliance (KE, US, GB, EU)
- Dividend and coupon reinvestment per target allocation
- Robo-to-human escalation on AUM threshold / distressed goal / profile mismatch
- Brinson-Hood-Beebower performance attribution (personalised benchmark)
- Client Lifetime Value (Gordon-growth model) and logistic churn prediction
- Regulatory reporting: CMA Kenya, FCA UK, SEC US
- Multi-currency valuation (KES, USD, EUR, GBP, NGN, GHS, ZAR)
- Factor exposure analysis: Fama-French 5-factor loadings
- Behavioural bias detection (loss aversion, recency bias, overconfidence)
- ESG portfolio filter, fee transparency report (MiFID II / CMA)
- Bulk onboarding, scheduled rebalance batch, portfolio health score

## Quick Start

```python
import asyncio
from capabilities.fintech.robo.service import RoboAdvisoryService

svc = RoboAdvisoryService(tenant_id="acme", actor_id="adviser-1")

async def main():
    # 1. Score a risk questionnaire
    q = await svc.risk_questionnaire("cust-001", {
        "investment_horizon": "5_to_10yr",
        "loss_reaction": "hold",
        "income_stability": "stable",
        "prior_experience": "moderate",
        "savings_rate": "medium",
    })
    profile_result = await svc.determine_risk_profile(q["questionnaire_id"])

    # 2. Onboard the client end-to-end
    onboarding = await svc.onboard_client("cust-001", {
        "risk_profile": profile_result["risk_profile"],
        "initial_investment_usd": 25_000,
        "monthly_contribution_usd": 500,
        "goal_type": "retirement",
        "target_amount_usd": 500_000,
        "horizon_years": 20,
        "currency": "USD",
    })

    # 3. Monte Carlo retirement projection
    sim = await svc.monte_carlo_retirement_simulation(
        onboarding["profile_id"], n_paths=10_000, contribution_monthly_usd=500
    )
    print(sim["probability_of_success"], sim["percentiles_usd"])

asyncio.run(main())
```

## Provides

| Service | Description |
|---------|-------------|
| robo_investor_profile_workflow | Create investor profiles with KYC, suitability, and supported risk profile |
| robo_goal_plan_workflow | Define investment goals with type, target amount, currency, and horizon |
| robo_model_portfolio_workflow | Publish model portfolios with risk profile and exact 100% allocation |
| robo_recommendation_workflow | Generate and approve investment recommendations with analysis evidence |
| robo_automation_workflow | Configure automated investment plans with approved recommendations and funding source |
| robo_drift_workflow | Record portfolio drift with analysis evidence |
| robo_tax_loss_workflow | Record tax-loss harvesting candidates with tax lot and positive loss amount |
| robo_review_workflow | Governance reviews for models, recommendations, and compliance |
| robo_agent_workflow | Register AI agents for suitability review, model review, and drift monitoring |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Client and adviser notifications |
| nlpc | NLP for recommendation narrative |
| keym | Key management |
| fintech_wealth | Client profile and mandate context |
| fintech_kyc | Investor identity verification |
| fintech_aml | AML screening |
| fintech_fraud | Fraud risk context |
| bia | Analytics and performance data |
| fin_rpt | Reporting |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| profiles.supported_risk_profiles | list | conservative, balanced, growth, aggressive | Risk tolerance categories |
| goals.supported_goal_types | list | retirement, education, home, wealth_growth, income, emergency | Goal categories |
| goals.supported_currencies | list | USD, KES, EUR, GBP, NGN, GHS, ZAR | Goal currencies |
| models.allocation_total_percent | number | 100 | Required model allocation total |
| automation.supported_cadences | list | one_time, weekly, monthly, quarterly | Investment cadences |
| drift.threshold_bps | number | 500 | Drift threshold in basis points |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-robo/dashboard | GET | fintech_robo:view | Overview |
| profiles | /fintech-robo/profiles | GET/POST | fintech_robo:profiles | Investors |
| goals | /fintech-robo/goals | GET/POST | fintech_robo:goals | Investors |
| models | /fintech-robo/models | GET/POST | fintech_robo:models | Models |
| recommendations | /fintech-robo/recommendations | GET/POST | fintech_robo:recommendations | Advice |
| automation | /fintech-robo/automation | GET/POST | fintech_robo:automation | Advice |
| drift | /fintech-robo/drift | GET/POST | fintech_robo:drift | Operations |
| tax_loss | /fintech-robo/tax-loss | GET/POST | fintech_robo:tax_loss | Operations |
| reviews | /fintech-robo/reviews | GET/POST | fintech_robo:reviews | Governance |
| agents | /fintech-robo/agents | GET/POST | fintech_robo:admin | Automation |
| settings | /fintech-robo/settings | GET/POST | fintech_robo:admin | Administration |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| profile_kyc_required | Investor profile without KYC | deny |
| profile_suitability_required | Profile without suitability evidence | deny |
| profile_risk_supported | Unsupported risk profile type | deny |
| goal_positive_target | Zero or negative goal target | deny |
| goal_horizon_required | Goal without investment horizon | deny |
| model_allocation_total | Model allocations do not sum to 100% | deny |
| model_policy_required | Model portfolio without policy reference | deny |
| recommendation_analysis_required | Recommendation without analysis | deny |
| recommendation_approval_required | Approving recommendation without reviewer | deny |
| automation_recommendation_required | Automation plan without approved recommendation | deny |
| automation_funding_source_required | Automation plan without funding source | deny |
| drift_analysis_required | Drift record without analysis | deny |
| tax_positive_loss | Tax-loss candidate with zero or negative loss | deny |
| robo_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_robo_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| InvestorProfile | id, client_reference, kyc_reference, suitability_reference, risk_profile, status |
| GoalPlan | id, profile_id, goal_type, target_amount, currency, horizon, status |
| ModelPortfolio | id, name, risk_profile, allocations, policy_reference, status |
| Recommendation | id, profile_id, goal_id, model_id, analysis_reference, status, reviewer_id |
| AutomationPlan | id, recommendation_id, cadence, funding_source_reference, status |
| DriftRecord | id, profile_id, drift_bps, analysis_reference |
| TaxLossCandidate | id, profile_id, tax_lot_reference, loss_amount |

## Streaming Events

Events emitted to the fintech event stream via Bytewax.

| Event | Trigger |
|-------|---------|
| investor_profile_created | Profile created |
| goal_plan_defined | Goal plan defined |
| model_portfolio_published | Model portfolio published |
| recommendation_generated | Recommendation generated |
| recommendation_approved | Recommendation approved |
| automation_plan_configured | Auto-invest plan configured |
| drift_recorded | Drift recorded |
| tax_loss_candidate_recorded | Tax-loss candidate recorded |
| robo_review_recorded | Review completed |
| robo_agent_registered | AI agent registered |

## World-Class Enhancements (v2.0)

1. **Monte Carlo Retirement Simulation** — 10,000 Gaussian paths per asset class; returns P10/P25/P50/P75/P90 percentile fan and probability-of-success metric. Replaces the deterministic single-path projection.

2. **Dynamic Lifecycle Glide Path** — `equity_pct = max(20, 110 - age)` scaled by years-to-goal; shifts allocation from equity-heavy to bond-heavy automatically as investor ages.

3. **Factor-Based Portfolio Construction** — Tracks Fama-French 5-factor loadings (Value, Momentum, Quality, Low-Vol, Size) per portfolio; detects factor drift alongside allocation drift.

4. **Market Regime Detection** — 2-state Hidden Markov Model (bull/bear) on rolling 12-month windows; adjusts expected return assumptions dynamically in bear regime.

5. **Tax-Lot Harvesting with Wash-Sale Compliance** — Per-lot tracking with 30-day wash-sale enforcement; multi-jurisdiction CGT rates (KE 15%, US 20%, GB 20%, EU 25%); returns wash-sale-safe replacement instruments.

6. **Behavioural Bias Nudge Engine** — Detects loss aversion, recency bias, and overconfidence from questionnaire response patterns; surfaces evidence-based nudge messages.

7. **Multi-Currency Portfolio Valuation** — FX-adjusted holdings reporting in all 7 supported currencies; FX PnL attribution; currency hedge recommendations.

8. **Robo-to-Human Escalation Workflow** — Trigger conditions: AUM > $500k, distressed goal (< 30% funded, < 2yr), aggressive profile + high AUM. On trigger: freezes automation, assigns human adviser.

9. **Drawdown Circuit Breaker** — Suspends all automation plans when portfolio drawdown from peak exceeds threshold (default 15%). Prevents unaware automated buying during crashes.

10. **Personalised Benchmark & BHB Attribution** — Policy portfolio constructed from investor's target allocation at period start; Brinson-Hood-Beebower decomposition: allocation, selection, and interaction effects per asset class.

11. **Regulatory Reporting Automation** — Structured report generation for CMA Kenya (Form RA-01), FCA UK (RMAR Section J), SEC US (Form ADV Part 2A); auto-file or signed PDF via Ollama.

12. **Goal Sensitivity Analysis** — 3×3×3 parameter grid: return ± 2 pp, contribution ± 25%, horizon ± 2 yr; 27-scenario matrix with goal-achievement flags; identifies highest-leverage intervention lever.

13. **Portfolio Stress Testing** — Historical scenarios: GFC 2008 (equities −50%), COVID 2020 (equities −35%), Rate Shock 2022 (bonds −20%); returns drawdown pct and estimated recovery months; flags suitability breach.

14. **Dividend and Coupon Reinvestment** — Tracks income events per asset class (explicit or yield-estimated); reinvests per target allocation; separates income from capital for tax reporting.

15. **Client Lifetime Value and Churn Prediction** — CLV via Gordon-growth formula (`AUM × fee × (1-(1+g)^-T) / (r-g)`); logistic churn score from 4 behavioural signals; routes to retention intervention (adviser call, goal email, fee discount).

## New Methods

### Monte Carlo simulation

```python
result = await svc.monte_carlo_retirement_simulation(
    profile_id="prof-abc",
    n_paths=10_000,
    contribution_monthly_usd=500.0,
)
# result["probability_of_success"]  -> 0.78
# result["percentiles_usd"]         -> {"p10": 180000, "p50": 310000, "p90": 510000}
```

### Goal sensitivity analysis

```python
sensitivity = await svc.goal_sensitivity_analysis(goal_id="goal-xyz")
# sensitivity["highest_leverage_lever"]  -> "contribution"
# sensitivity["overall_success_rate"]    -> 0.63
# sensitivity["grid"]                    -> list of 27 scenario dicts
```

### Portfolio stress test

```python
stress = await svc.portfolio_stress_test(profile_id="prof-abc")
# stress["worst_case"]["scenario"]          -> "GFC_2008"
# stress["worst_case"]["drawdown_pct"]      -> 38.5
# stress["worst_case"]["estimated_recovery_months"] -> 28.4
```

### Drawdown circuit breaker

```python
cb = await svc.drawdown_circuit_breaker_check(
    portfolio_id="prof-abc",
    peak_value_usd=95_000.0,
    drawdown_threshold_pct=15.0,
)
# cb["circuit_open"]  -> True / False
# cb["action"]        -> "automation_suspended" | "automation_continues"
```

### Churn prediction and CLV

```python
churn = await svc.churn_probability("cust-001")
# churn["churn_risk_tier"]             -> "high" | "medium" | "low"
# churn["recommended_intervention"]    -> "immediate_adviser_call"

clv = await svc.client_lifetime_value("cust-001")
# clv["lifetime_value_usd"]    -> 3_420.50
# clv["annual_fee_revenue_usd"] -> 137.50
```

## Key Service Methods

### Core Advisory

- `describe()` / `evaluate()` — capability contract and policy enforcement
- `risk_questionnaire(customer_id, responses)` — 5-dimension questionnaire scoring
- `determine_risk_profile(questionnaire_id)` — finalise risk profile with allocation and return estimate
- `recommended_portfolio(risk_profile, amount)` — model portfolio with projected value and fees
- `onboard_client(customer_id, plan)` — full onboarding: profile + goal + automation setup

### Automation and Rebalancing

- `auto_invest(customer_id, amount, frequency)` — recurring investment per model portfolio
- `auto_rebalance(portfolio_id, ...)` — drift-threshold-guarded rebalance with dry-run mode
- `drawdown_circuit_breaker_check(portfolio_id, peak_value_usd)` — suspend automation on excessive drawdown
- `scheduled_rebalance_batch(cadence)` — batch rebalance all portfolios

### Quantitative Analysis

- `monte_carlo_retirement_simulation(profile_id, n_paths)` — probabilistic retirement projection
- `compute_glide_path_allocation(profile_id, goal_id, current_age)` — lifecycle allocation glide
- `portfolio_stress_test(profile_id, scenarios)` — historical scenario stress testing
- `goal_sensitivity_analysis(goal_id)` — 3D sensitivity grid

### Tax and Income

- `tax_optimisation(portfolio_id, jurisdiction)` — TLH candidates with tax saving estimates
- `tax_lot_harvesting_engine(profile_id, jurisdiction)` — wash-sale-safe lot-level harvesting
- `reinvest_income(portfolio_id, income_events)` — dividend/coupon reinvestment

### Client Analytics

- `client_lifetime_value(customer_id)` — Gordon-growth CLV
- `churn_probability(customer_id)` — logistic churn score with intervention routing
- `evaluate_escalation_triggers(profile_id)` — robo-to-human escalation decision
- `portfolio_health_score(profile_id)` — 0–100 composite health (diversity, drift, goals)

## Edge Cases Handled

- Model allocation totals must equal exactly 100% — rounding to 99.99% or 100.01% is rejected
- Automation plans require an approved (not just generated) recommendation
- Tax-loss candidates require a positive loss amount — zero or negative losses are rejected
- Drift is recorded in basis points (100 bps = 1%); threshold is configurable (default 500 bps)
- Suitability evidence is separate from KYC — both are required before a profile can be created
- Circuit breaker evaluates drawdown from the supplied `peak_value_usd`; it does not maintain a high-water mark internally — callers must supply the peak

## Composability

- **Upstream**: `fintech_wealth` provides the client profile context; market data feeds (declared as `market_data` adapter) provide price data for drift calculations
- **Downstream**: `fintech_portfolio` uses robo model portfolios as allocation templates; `fintech_trading` executes orders generated from automation plans
- **Peer**: Deployed alongside `fintech_wealth` (human advisory layer) and `fintech_portfolio` (portfolio book management)

## Development Notes

- `SUPPORTED_REVIEW_STATUSES` for robo is `["approved", "rejected", "needs_changes"]` — `escalated` is intentionally absent; escalation is handled via `evaluate_escalation_triggers`, not the review workflow
- `market_data` is a runtime soft dependency; the capability functions without live market data using cached or manual valuations
- Tax-loss harvesting records are advisory — actual trade execution happens in `fintech_trading`
- Monte Carlo uses a seeded `random.Random(42)` for reproducibility in tests; pass `n_paths=100` in unit tests for speed

© 2025 Datacraft — www.datacraft.co.ke | nyimbi@gmail.com
