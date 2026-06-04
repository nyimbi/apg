# Wealth Management

## Overview
Wealth Management provides regulated advisory and portfolio services: client profile onboarding with KYC, tax, and risk evidence; suitability assessment across risk tolerance, investment horizon, and goals; portfolio creation with advisor assignment and investment policy statement; advisory mandate setup (advisory, discretionary, model, execution-only); portfolio rebalance proposals with exact 100% allocation totals and analysis evidence; trade order staging with approval gates for large orders; performance recording; and fee schedule management. It is the client-facing wealth services layer that backs Robo Advisory and Portfolio Management.

Rebalance allocations must total exactly 100%. Mandates must match their portfolio. Large orders require human approval. Fee percentages are bounded 0–100%. All wealth management events stream to `apg.fintech.wealth.lifecycle` via Bytewax.

## Capability ID
`fintech_wealth`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| wealth_client_profile_workflow | Register client profiles with KYC, tax profile, and risk evidence |
| suitability_profile_workflow | Capture suitability profiles with risk tolerance, horizon, and investment goals |
| portfolio_management_workflow | Create portfolios with currency, advisor, and investment policy statement |
| advisory_mandate_workflow | Create advisory, discretionary, model, and execution-only mandates |
| portfolio_rebalance_workflow | Propose rebalances with exact 100% allocation and analysis evidence |
| wealth_order_workflow | Stage buy/sell/switch orders with risk reference and large-order approval |
| performance_reporting_workflow | Record performance snapshots with valuation and benchmark |
| wealth_fee_workflow | Record fee schedules with bounded percentage and fee contract |
| wealth_agent_workflow | Register AI agents for advisor review, suitability, portfolio, and fee review |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Client and adviser notifications |
| nlpc | NLP for mandate and report narrative |
| keym | Key management |
| fintech_kyc | Client identity verification |
| fintech_aml | AML screening |
| fintech_fraud | Fraud risk context |
| fintech_payments | Order settlement execution |
| fintech_wallets | Wallet-based cash management |
| bia | Performance analytics |
| fin_rpt | Wealth reporting |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| suitability.supported_risk_profiles | list | conservative, balanced, growth, aggressive | Risk tolerance categories |
| suitability.supported_tolerances | list | low, medium, high | Risk tolerance levels |
| suitability.supported_horizons | list | one_year, three_years, five_years, ten_years, retirement | Investment horizons |
| mandates.supported_types | list | advisory, discretionary, model, execution_only | Mandate types |
| orders.supported_sides | list | buy, sell, switch | Order directions |
| orders.large_order_threshold_minor | number | 10000000 | Large order requiring approval (minor units) |
| fees.minimum_percent | number | 0 | Minimum fee percentage |
| fees.maximum_percent | number | 100 | Maximum fee percentage |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-wealth/dashboard | GET | fintech_wealth:view | Overview |
| clients | /fintech-wealth/clients | GET/POST | fintech_wealth:clients | Clients |
| suitability | /fintech-wealth/suitability | GET/POST | fintech_wealth:suitability | Clients |
| portfolios | /fintech-wealth/portfolios | GET/POST | fintech_wealth:portfolios | Portfolios |
| mandates | /fintech-wealth/mandates | GET/POST | fintech_wealth:mandates | Portfolios |
| rebalances | /fintech-wealth/rebalances | GET/POST | fintech_wealth:rebalances | Portfolios |
| orders | /fintech-wealth/orders | GET/POST | fintech_wealth:orders | Trading |
| performance | /fintech-wealth/performance | GET/POST | fintech_wealth:performance | Reporting |
| fees | /fintech-wealth/fees | GET/POST | fintech_wealth:fees | Operations |
| agents | /fintech-wealth/agents | GET/POST | fintech_wealth:admin | Automation |
| settings | /fintech-wealth/settings | GET/POST | fintech_wealth:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| client_kyc_required | Client without KYC evidence | deny |
| client_tax_required | Client without tax profile | deny |
| client_risk_required | Client without risk evidence | deny |
| suitability_goals_required | Suitability without investment goals | deny |
| portfolio_advisor_required | Portfolio without assigned advisor | deny |
| portfolio_policy_required | Portfolio without investment policy statement | deny |
| mandate_suitability_required | Mandate without suitability profile | deny |
| mandate_type_supported | Unsupported mandate type | deny |
| rebalance_mandate_matches_portfolio | Rebalance mandate belongs to different portfolio | deny |
| rebalance_allocation_total | Allocations do not sum to 100% | deny |
| rebalance_analysis_required | Rebalance without analysis evidence | deny |
| large_order_requires_approval | Order > threshold without human approval | require_review |
| performance_valuation_required | Performance without valuation | deny |
| performance_benchmark_required | Performance without benchmark | deny |
| fee_percent_bounded | Fee outside 0–100% range | deny |
| fee_contract_required | Fee schedule without fee contract | deny |
| wealth_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_wealth_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| WealthClient | id, name, kyc_reference, tax_reference, risk_reference, status |
| SuitabilityProfile | id, client_id, risk_profile, tolerance, horizon, goals, status |
| WealthPortfolio | id, client_id, name, currency, advisor_id, investment_policy_reference, status |
| AdvisoryMandate | id, portfolio_id, suitability_id, mandate_type, policy_reference, status |
| RebalanceProposal | id, portfolio_id, mandate_id, allocations, analysis_reference, status |
| WealthOrder | id, portfolio_id, instrument_reference, side, quantity, risk_reference, human_approval_reference, status |
| PerformanceRecord | id, portfolio_id, valuation_reference, benchmark_reference, period |
| FeeSchedule | id, portfolio_id, fee_percent, fee_contract_reference |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| client_profile_registered | Client profile created |
| suitability_profile_captured | Suitability assessment recorded |
| portfolio_created | Portfolio created |
| advisory_mandate_created | Mandate established |
| rebalance_proposed | Rebalance proposal submitted |
| order_staged | Trade order staged |
| performance_recorded | Performance snapshot recorded |
| fee_schedule_recorded | Fee schedule recorded |
| wealth_agent_registered | AI agent registered |

## Edge Cases Handled
- Rebalance mandate must match the portfolio — a mandate from a different portfolio cannot be used to rebalance a given portfolio; this prevents cross-portfolio allocation changes
- Tax profile is a separate requirement from KYC — a client with a completed KYC profile still requires a tax profile before wealth services can be rendered; this covers tax residency, withholding rates, and reporting obligations
- Investment goals are required at suitability capture — without goals, the suitability assessment is incomplete and cannot be used for mandate or recommendation creation
- Large order threshold is expressed in minor units (10,000,000 minor units = USD 100,000 or KES 100,000 depending on currency); the `large_order` context flag is set by the service layer before rule evaluation
- Fee percentage of exactly 0% is valid (pro-bono wealth management); fee percentage of exactly 100% is valid but unusual; both are accepted as they fall within the [0, 100] closed interval

## Composability
- **Upstream**: `fintech_kyc`, `fintech_aml`, and `fintech_fraud` provide client onboarding evidence; market data adapters (referenced via `bia`) provide valuation and benchmark data for performance recording
- **Downstream**: `fintech_robo` builds on Wealth Management client profiles and mandates for automated advice; `fintech_portfolio` uses wealth portfolio books for institutional-grade portfolio operations; `fintech_trading` executes orders staged by wealth managers
- **Peer**: Deployed alongside `fintech_robo` (automated advisory) and `fintech_portfolio` (portfolio book management) in a full wealth management stack

## Development Notes
- `execution_only` mandate type means the advisor does not provide advice — the client makes their own investment decisions; the capability still requires a suitability profile as a documented acknowledgment of the execution-only arrangement
- Wealth Management uses `fin_rpt` for regulatory reporting (MiFID II suitability reports, performance reports) — these are distinct from the internal performance records stored in the capability
- The `rebalance_allocation_total` rule requires exact 100% equality; the service layer should normalize allocations before setting the `allocation_totals_100` flag to avoid floating-point rounding failures
- `switch` order side covers switching between funds within the same portfolio — it is semantically different from a sell+buy pair; it enables in-specie transfers without cash settlement
