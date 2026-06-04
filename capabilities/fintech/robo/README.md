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
