# Robo Advisory

## Overview
Robo Advisory provides algorithm-guided investment advice under governance: investor profile creation with KYC and suitability evidence, goal planning, model portfolio publication with exact 100% allocation totals, recommendation generation and approval workflows, automated investment plan configuration, portfolio drift monitoring, tax-loss harvesting candidate recording, and governance reviews. It builds on Wealth Management by making model-driven recommendations, automated rebalancing, and tax optimization first-class governed operations.

Recommendations require an approved model and analysis before generation. Automation plans require an approved recommendation. Model allocations must total exactly 100%. All robo advisory events stream to `apg.fintech.robo.lifecycle` via Bytewax.

## Capability ID
`fintech_robo`  Version: 1.1.0

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

## Edge Cases Handled
- Model allocation totals must equal exactly 100% — rounding to 99.99% or 100.01% is rejected; the `allocation_totals_100` flag requires exact equality computed by the service layer
- Automation plans require an approved (not just generated) recommendation — a recommendation in `generated` status cannot trigger automation; the status must be `approved`
- Tax-loss candidates require a positive loss amount — a candidate with zero loss has no harvesting value and is rejected; negative losses (gains) are nonsensical and also rejected
- Drift is recorded in basis points (100 bps = 1%); the drift threshold (500 bps = 5%) is configurable; the rule engine does not enforce the threshold — it only checks that an analysis reference is present
- Suitability evidence is separate from KYC — a KYC-verified investor still requires a separate suitability assessment (risk questionnaire outcome) before a profile can be created

## Composability
- **Upstream**: `fintech_wealth` provides the client profile context that robo profiles link to; market data feeds (referenced as `market_data` adapter) provide price data for drift calculations
- **Downstream**: `fintech_portfolio` uses robo model portfolios as the allocation templates for managed portfolio books; `fintech_trading` executes orders generated from automation plans
- **Peer**: Deployed alongside `fintech_wealth` (human advisory layer) and `fintech_portfolio` (portfolio book management)

## Development Notes
- `SUPPORTED_REVIEW_STATUSES` for robo is `["approved", "rejected", "needs_changes"]` — notably missing `escalated` compared to other capabilities; robo reviews are expected to be resolved at the adviser level
- `market_data` is declared as an adapter in `DEFAULT_CONFIGURATION` but not in `REQUIRES` — it is a runtime soft dependency; the capability can function without live market data (using cached or manual valuations)
- The drift threshold (500 bps) is a configuration default; individual profiles can have different thresholds if the service layer implements per-profile overrides
- Tax-loss harvesting records are advisory — the capability records candidates for tax-loss harvesting; the actual trade execution happens in `fintech_trading`; there is no direct link between a tax-loss record and a trade order in the rule engine

## New Features (v2.0.0)

### Monte Carlo Simulation
`monte_carlo_retirement_simulation(profile_id, n_paths=10000)` runs 10,000 Gaussian return paths per asset class and returns a P10/P25/P50/P75/P90 percentile fan plus a probability-of-success metric. Replaces the single deterministic compound-growth projection.

### Lifecycle Glide Path
`compute_glide_path_allocation(profile_id, goal_id, current_age)` computes a target-date allocation using `equity_pct = max(20, 110 - current_age)` scaled by years-to-goal. Returns delta vs current allocation and a rebalance flag.

### Portfolio Stress Testing
`portfolio_stress_test(profile_id, scenarios)` applies three built-in historical scenarios (GFC 2008, COVID 2020, Rate Shock 2022) plus any custom shocks. Returns drawdown percentage and estimated recovery months per scenario.

### Goal Sensitivity Analysis
`goal_sensitivity_analysis(goal_id)` runs a 3×3×3 grid over return ± 2 pp, monthly contribution ± 25%, and horizon ± 2 years. Returns a 27-scenario matrix with goal-achievement flags and identifies the highest-leverage intervention lever.

### Drawdown Circuit Breaker
`drawdown_circuit_breaker_check(portfolio_id, peak_value_usd, drawdown_threshold_pct=15)` suspends automation plans when portfolio drawdown from peak exceeds the threshold. Prevents automated buying during crash conditions without client awareness.

### Robo-to-Human Escalation
`evaluate_escalation_triggers(profile_id)` evaluates AUM threshold ($500k), distressed goals (< 30% funded, < 2yr horizon), and risk-profile mismatch. Returns escalation decision and freezes automation plans pending human review.

### Tax-Lot Harvesting Engine
`tax_lot_harvesting_engine(profile_id, jurisdiction, min_lot_age_days=31)` identifies wash-sale-safe harvest candidates per tax lot with multi-jurisdiction CGT rates (KE 15%, US 20%, GB 20%, EU 25%). Returns replacement instrument recommendations.

### Income Reinvestment
`reinvest_income(portfolio_id, income_events)` credits dividend and coupon income (from explicit events or yield estimates) and reinvests per target allocation. Separates income from capital for tax reporting.

### Client Lifetime Value
`client_lifetime_value(customer_id)` uses a Gordon-growth CLV formula: `AUM × fee × (1-(1+g)^-T) / (r-g)`. Returns annual fee revenue and lifetime value estimate.

### Churn Prediction
`churn_probability(customer_id)` computes a logistic churn score from four behavioural signals (goal progress, portfolio health, tenure, AUM tier) and routes to the appropriate retention intervention (adviser call, goal progress email, fee discount).

## Key Service Methods

### Core Advisory
- `describe()` / `evaluate()` — capability contract and policy enforcement
- `risk_questionnaire(customer_id, responses)` — 5-dimension questionnaire scoring
- `determine_risk_profile(questionnaire_id)` — finalise risk profile with allocation and return estimate
- `recommended_portfolio(risk_profile, amount)` — model portfolio with projected value and fees
- `onboard_client(customer_id, plan)` — full onboarding: profile + goal + automation setup

### Automation & Rebalancing
- `auto_invest(customer_id, amount, frequency)` — recurring investment per model portfolio
- `auto_rebalance(portfolio_id, ...)` — drift-threshold-guarded rebalance with dry-run mode
- `drawdown_circuit_breaker_check(portfolio_id, peak_value_usd)` — suspend automation on excessive drawdown

### Quantitative Analysis
- `monte_carlo_retirement_simulation(profile_id, n_paths)` — probabilistic retirement projection
- `compute_glide_path_allocation(profile_id, goal_id, current_age)` — lifecycle allocation glide
- `portfolio_stress_test(profile_id, scenarios)` — historical scenario stress testing
- `goal_sensitivity_analysis(goal_id)` — 3D sensitivity grid

### Tax & Income
- `tax_optimisation(portfolio_id, jurisdiction)` — TLH candidates with tax saving estimates
- `tax_lot_harvesting_engine(profile_id, jurisdiction)` — wash-sale-safe lot-level harvesting
- `reinvest_income(portfolio_id, income_events)` — dividend/coupon reinvestment

### Client Analytics
- `client_lifetime_value(customer_id)` — Gordon-growth CLV
- `churn_probability(customer_id)` — logistic churn score with intervention routing
- `evaluate_escalation_triggers(profile_id)` — robo-to-human escalation decision

_(See `service.py` for all signatures and `docs/user_guide.md` for usage examples.)_
