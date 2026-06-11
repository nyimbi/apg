# Portfolio Management

## Overview
Portfolio Management provides regulated investment book operations: portfolio book creation, holding ledger recording, allocation policy activation (totals must equal exactly 100%), valuation capture, benchmark assignment, risk exposure tracking, performance attribution, cash movement recording, corporate action processing, compliance breach recording, and governance reviews. It is the investment operations layer for discretionary, advisory, model, and execution-only portfolios.

Allocation policies must total exactly 100% before activation. Valuations require a source and valuation date. Performance attribution requires a benchmark. All portfolio lifecycle events stream to `apg.fintech.portfolio.lifecycle` via Bytewax.

## Capability ID
`fintech_portfolio`  Version: 2.0.0

## Provides
| Service | Description |
|---------|-------------|
| portfolio_book_workflow | Create portfolio books with type, base currency, owner, and investment policy |
| portfolio_holding_workflow | Record holdings with instrument, positive quantity, and positive cost |
| portfolio_allocation_policy_workflow | Activate allocation policies with exact 100% total and policy reference |
| portfolio_valuation_workflow | Record valuations with positive market value, source, and valuation date |
| portfolio_benchmark_workflow | Assign benchmark indices with policy reference |
| portfolio_risk_workflow | Record risk exposures with source, as-of date, and limit reference |
| portfolio_attribution_workflow | Record performance attribution with period, source, and benchmark |
| portfolio_cash_workflow | Record cash movements with amount, currency, and reference |
| portfolio_corporate_action_workflow | Record dividends, splits, mergers, coupons, and redemptions with evidence |
| portfolio_compliance_workflow | Record and review compliance breaches with severity controls |
| portfolio_review_workflow | Governance reviews for allocations, valuations, and compliance |
| portfolio_agent_workflow | Register AI agents for book review, valuation, risk exposure, and attribution |
| portfolio_twr_workflow | GIPS-compliant time-weighted return with sub-period chain-linking |
| portfolio_mwr_workflow | Money-weighted return (IRR), MOIC, and DPI for closed-end funds |
| portfolio_stress_test_workflow | Multi-scenario stress testing with per-asset-class shock factors |
| portfolio_counterparty_workflow | Single-counterparty concentration risk aggregation across portfolios |
| portfolio_fx_workflow | FX rate store for multi-currency holding revaluation |
| portfolio_clone_workflow | Clone model/template portfolio to a new client book |
| portfolio_audit_query_workflow | Query and export the structured audit event log |
| portfolio_client_report_workflow | Assemble structured client-facing performance reports (IPS, factsheet) |
| portfolio_esg_workflow | Weighted ESG scoring and exclusion breach detection |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Portfolio operations notifications |
| nlpc | NLP processing |
| keym | Key management |
| fintech_wealth | Wealth management client context |
| fintech_robo | Robo advisory model portfolios |
| fintech_payments | Cash movement execution |
| fintech_wallets | Wallet-based cash management |
| fintech_kyc | Investor identity |
| fintech_aml | AML screening |
| fintech_fraud | Fraud risk context |
| bia | Analytics and reporting |
| fin_rpt | Financial reporting |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| portfolios.supported_types | list | discretionary, advisory, model, execution_only, treasury | Portfolio management styles |
| portfolios.supported_currencies | list | USD, KES, EUR, GBP, NGN, GHS, ZAR | Base currencies |
| allocation_policies.allocation_total_percent | number | 100 | Required allocation total |
| corporate_actions.supported_types | list | dividend, split, merger, spin_off, rights_issue, coupon, redemption | Corporate action types |
| compliance.supported_severities | list | low, medium, high, critical | Breach severity levels |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-portfolio/dashboard | GET | fintech_portfolio:view | Overview |
| portfolios | /fintech-portfolio/portfolios | GET/POST | fintech_portfolio:portfolios | Books |
| holdings | /fintech-portfolio/holdings | GET/POST | fintech_portfolio:holdings | Books |
| allocations | /fintech-portfolio/allocations | GET/POST | fintech_portfolio:allocations | Policy |
| valuations | /fintech-portfolio/valuations | GET/POST | fintech_portfolio:valuations | Operations |
| benchmarks | /fintech-portfolio/benchmarks | GET/POST | fintech_portfolio:benchmarks | Policy |
| risk | /fintech-portfolio/risk | GET/POST | fintech_portfolio:risk | Risk |
| attribution | /fintech-portfolio/attribution | GET/POST | fintech_portfolio:attribution | Performance |
| cash | /fintech-portfolio/cash | GET/POST | fintech_portfolio:cash | Operations |
| corporate_actions | /fintech-portfolio/corporate-actions | GET/POST | fintech_portfolio:corporate_actions | Operations |
| compliance | /fintech-portfolio/compliance | GET/POST | fintech_portfolio:compliance | Governance |
| reviews | /fintech-portfolio/reviews | GET/POST | fintech_portfolio:reviews | Governance |
| agents | /fintech-portfolio/agents | GET/POST | fintech_portfolio:admin | Automation |
| settings | /fintech-portfolio/settings | GET/POST | fintech_portfolio:admin | Administration |
| twr | /fintech-portfolio/twr | POST | fintech_portfolio:performance | Performance |
| mwr | /fintech-portfolio/mwr | POST | fintech_portfolio:performance | Performance |
| stress_test | /fintech-portfolio/stress-test | POST | fintech_portfolio:risk | Risk |
| counterparty | /fintech-portfolio/counterparty-exposure | GET | fintech_portfolio:risk | Risk |
| fx_rates | /fintech-portfolio/fx-rates | GET/POST | fintech_portfolio:operations | Operations |
| clone | /fintech-portfolio/clone | POST | fintech_portfolio:admin | Administration |
| audit_query | /fintech-portfolio/audit | GET | fintech_portfolio:admin | Administration |
| client_report | /fintech-portfolio/client-report | POST | fintech_portfolio:view | Reports |
| esg | /fintech-portfolio/esg | GET/POST | fintech_portfolio:view | ESG |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| portfolio_type_supported | Unsupported portfolio type | deny |
| holding_positive_quantity | Zero or negative holding quantity | deny |
| holding_positive_cost | Zero or negative holding cost | deny |
| allocation_total_required | Allocations do not sum to 100% | deny |
| allocation_policy_reference_required | Activation without policy reference | deny |
| valuation_positive_market_value | Zero or negative market value | deny |
| valuation_date_required | Valuation without date | deny |
| valuation_source_required | Valuation without source reference | deny |
| risk_source_required | Risk exposure without source | deny |
| risk_as_of_date_required | Risk exposure without as-of date | deny |
| attribution_period_required | Attribution without period | deny |
| corporate_action_evidence_required | Corporate action without evidence | deny |
| portfolio_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_portfolio_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| PortfolioBook | id, owner_id, name, portfolio_type, base_currency, investment_policy_reference, status |
| PortfolioHolding | id, portfolio_id, instrument_reference, quantity, cost, currency |
| AllocationPolicy | id, portfolio_id, allocations, policy_reference, status |
| PortfolioValuation | id, portfolio_id, market_value, currency, valuation_date, source_reference |
| BenchmarkAssignment | id, portfolio_id, index_reference, policy_reference |
| RiskExposure | id, portfolio_id, metric, amount, limit_reference, source_reference, as_of_date |
| PerformanceAttribution | id, portfolio_id, period, source_reference, benchmark_reference, contributions |
| PortfolioCash | id, portfolio_id, amount, currency, reference |
| CorporateAction | id, instrument_reference, action_type, effective_date, evidence_reference |
| ComplianceBreach | id, portfolio_id, severity, evidence_reference, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| portfolio_book_created | Portfolio book created |
| portfolio_holding_recorded | Holding recorded |
| allocation_policy_activated | Allocation policy activated |
| portfolio_valuation_recorded | Valuation recorded |
| benchmark_assigned | Benchmark assigned |
| risk_exposure_recorded | Risk exposure recorded |
| performance_attribution_recorded | Attribution recorded |
| cash_movement_recorded | Cash movement recorded |
| corporate_action_recorded | Corporate action processed |
| compliance_breach_recorded | Breach recorded |
| portfolio_review_recorded | Review completed |
| portfolio_agent_registered | AI agent registered |

## Edge Cases Handled
- Allocation totals must equal exactly 100% — rounding errors (e.g., 99.99%) are not tolerated; the `allocation_totals_100` flag must be set by the service layer after verifying exact equality
- Valuations with zero market value are denied — a portfolio with zero market value is either empty or erroneously valued; the rule forces explicit handling rather than silent acceptance
- Corporate actions apply to an instrument, not a portfolio — the same dividend or split can affect holdings across multiple portfolios; the action is recorded at the instrument level with an effective date
- Risk exposure as-of-date is required to prevent stale exposure records being confused with current positions
- Holdings can have fractional quantities (ETF fractional shares) but cannot be zero or negative — the `positive_quantity` rule enforces strict positivity
- TWR requires at least two valuation records; fewer returns `insufficient_data` rather than a misleading 0.0
- MWR (IRR) annualisation uses the actual calendar distance between start and end date to avoid compounding artifacts on sub-annual periods
- Stress test scenarios without a matching instrument_id fall back to `equity` then `default` shock keys, ensuring portfolios with unmapped instruments are not silently excluded
- Counterparty concentration is computed only against holdings with an `issuer_id` attribute; unattributed holdings are grouped under `unattributed` and excluded from the limit check
- ESG score aggregation requires explicit `record_esg_rating` calls per instrument; unscored holdings are listed separately and do not dilute the weighted average
- Portfolio cloning copies the allocation policy from the source but starts with zero holdings, preventing unintended position duplication across client books

## Composability
- **Upstream**: `fintech_wealth` provides client profile and mandate context; `fintech_robo` provides model portfolio templates; market data feeds are adapter boundaries referenced by ID
- **Downstream**: `fintech_trading` consumes portfolio positions for order generation; `bia` and `fin_rpt` consume valuations, attribution, and risk data for reporting
- **Peer**: Deployed alongside `fintech_wealth` (client-facing advisory layer) and `fintech_trading` (execution layer)

## Development Notes
- `treasury` portfolio type is included alongside standard investment types — this supports corporate treasury portfolio management alongside client investment books
- Performance attribution `contributions` is a free-form dict at the model level; the rule engine does not validate the attribution methodology — that is the responsibility of the analytics adapter
- `market_data` is declared as an adapter in `DEFAULT_CONFIGURATION` but not in `REQUIRES` — it is a soft dependency accessed via the adapter reference at runtime
- The `fintech_robo` dependency links robo advisory model portfolios to discretionary/model portfolio books, enabling automated rebalancing signals
