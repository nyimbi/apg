# Portfolio Management

## Overview

Portfolio Management provides regulated investment book operations: portfolio book creation, holding ledger recording, allocation policy activation (totals must equal exactly 100%), valuation capture, benchmark assignment, risk exposure tracking, performance attribution, cash movement recording, corporate action processing, compliance breach recording, and governance reviews. It is the investment operations layer for discretionary, advisory, model, and execution-only portfolios.

Version 3.0.0 adds Barra-style factor risk decomposition, liquidity risk scoring (days-to-liquidate), glide path management for target-date funds, specific-lot tax tracking with Kenya CGT calculation, pre-trade compliance checking, risk budget monitoring, transaction cost analysis (TCA), household/sleeve consolidated views, DRIP (dividend reinvestment) automation, real-time NAV streaming, multi-level attribution, composite benchmark construction, portfolio scoring, and NATS-backed event-sourced audit.

Allocation policies must total exactly 100% before activation. Valuations require a source and valuation date. Performance attribution requires a benchmark. All portfolio lifecycle events stream to `apg.fintech.portfolio.lifecycle` via Bytewax. DRIP executions and risk budget breaches publish to NATS `apg.fintech.portfolio.*` subjects.

## Capability ID

`fintech_portfolio`  Version: 3.0.0

## Quick Start

```python
from apg_fintech_portfolio.service import PortfolioManagementService

svc = PortfolioManagementService(tenant_id="acme", nats_url="nats://localhost:4222")

# Create a portfolio book
book = await svc.create_portfolio(
    portfolio_id="pf-001",
    owner_id="client-42",
    name="Growth Fund",
    portfolio_type="discretionary",
    base_currency="KES",
    investment_policy_reference="IPS-2025-001",
)

# Record a holding
await svc.add_holding("pf-001", "SCOM.NSE", quantity=10_000, cost_minor=1_850_000, currency="KES")

# Record a valuation and compute TWR
await svc.portfolio_valuation("pf-001", market_value_minor=2_100_000, currency="KES",
                               valuation_date="2025-12-31", source_reference="NAV-CALC-01")
result = await svc.time_weighted_return("pf-001", start_date="2025-01-01", end_date="2025-12-31")
```

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
| portfolio_factor_risk_workflow | Barra-style MCTR factor risk decomposition |
| portfolio_liquidity_workflow | Days-to-liquidate scoring and ADV bucket classification |
| portfolio_glide_path_workflow | Target-date glide path registration and application |
| portfolio_tax_lot_workflow | Specific-lot FIFO/LIFO/highest-cost disposal with Kenya CGT |
| portfolio_pre_trade_workflow | Pre-trade compliance: prohibited list, concentration, mandate check |
| portfolio_risk_budget_workflow | Risk budget registration and utilisation monitoring |
| portfolio_tca_workflow | Transaction cost analysis: implementation shortfall and broker ranking |
| portfolio_household_workflow | Household/sleeve consolidated AUM, allocation, and ESG view |
| portfolio_drip_workflow | Dividend reinvestment automation with fractional unit support |

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
| factor_risk | /fintech-portfolio/factor-risk | GET/POST | fintech_portfolio:risk | Risk |
| liquidity | /fintech-portfolio/liquidity | GET/POST | fintech_portfolio:risk | Risk |
| glide_path | /fintech-portfolio/glide-path | GET/POST | fintech_portfolio:admin | Administration |
| tax_lots | /fintech-portfolio/tax-lots | GET/POST | fintech_portfolio:operations | Operations |
| pre_trade | /fintech-portfolio/pre-trade-check | POST | fintech_portfolio:compliance | Compliance |
| risk_budget | /fintech-portfolio/risk-budget | GET/POST | fintech_portfolio:risk | Risk |
| tca | /fintech-portfolio/tca | GET/POST | fintech_portfolio:performance | Performance |
| household | /fintech-portfolio/household | GET/POST | fintech_portfolio:view | Reports |
| drip | /fintech-portfolio/drip | GET/POST | fintech_portfolio:operations | Operations |
| nav_stream | /fintech-portfolio/{id}/nav-stream | GET (SSE) | fintech_portfolio:view | Streaming |

## World-Class Enhancements (v2.0)

15 institutional-grade capabilities added in v3.0.0, benchmarked against Bloomberg PORT, MSCI RiskMetrics, BlackRock Aladdin, and Charles River IMS:

| # | Enhancement | Category | Competitor Baseline |
|---|-------------|----------|---------------------|
| I1 | **Factor Risk Decomposition (Barra-style MCTR)** — per-holding marginal contribution to risk across equity_beta, duration_dv01, credit_spread_dv01, fx_delta, real_estate_beta; diversification ratio | Risk Analytics | MSCI RiskMetrics, Bloomberg PORT |
| I2 | **Liquidity Risk Scoring with Days-to-Liquidate** — ADV-based bucket classification (liquid/semi-liquid/illiquid/locked); % AUM liquidatable within 1/5/10/30 days; CMA LCR-compliant | Risk Analytics | BlackRock Aladdin, MSCI Liquidity Risk |
| I3 | **Automated Rebalancing Execution with Block Trading** — NATS JetStream order emission, PENDING_EXECUTION state tracking, fill reconciliation, portfolio allocation lock/unlock | Operations | Charles River IMS, SimCorp Dimension |
| I4 | **Target-Date Glide Path Management** — GlidePath waypoints, automatic de-risking via `apply_glide_path`, schedule with days-to-next and de-risking velocity | Product/Strategy | Vanguard Target Retirement, BlackRock LifePath |
| I5 | **Tax-Lot Tracking with Specific Identification** — FIFO/LIFO/highest-cost/specific-lot disposal, per-lot Kenya CGT (15%) calculation, holding period classification | Tax & Compliance | Advent Geneva, SS&C Eze Eclipse |
| I6 | **Real-Time NAV Streaming via NATS + Bytewax** — async generator subscribing to `apg.market_data.prices.>`, NavTick emission, SSE endpoint at `/nav-stream` | Infrastructure | Bloomberg B-PIPE, Refinitiv Eikon |
| I7 | **Multi-Portfolio Consolidated View (Household/Sleeve)** — total AUM, weighted allocation, blended ESG, Herfindahl concentration across all accounts for a beneficial owner | Client UX | Orion Portfolio Solutions, Addepar |
| I8 | **Pre-Trade Compliance Checking** — prohibited instrument list, post-trade concentration (10% AUM limit), mandate type alignment; auto-records ComplianceBreach on violations | Compliance | Charles River Compliance, Bloomberg AIM |
| I9 | **Transaction Cost Analysis (TCA)** — implementation shortfall per trade, VWAP slippage, Almgren-Chriss market impact, broker performance ranking; best-execution reporting | Performance | ITG/Virtu TCA, Bloomberg TCA |
| I10 | **Multi-Level Attribution (Asset Class / Sector / Geography)** — BHB allocation, selection, and interaction effects at each level; waterfall decomposition tree | Performance Analytics | FactSet PA, Bloomberg PORT Multi-Level |
| I11 | **Custom Composite Benchmark Construction** — constituent-weighted benchmark return from recorded index returns; active return vs composite; benchmark return decomposition | Performance | MSCI Custom Index Builder, FTSE Russell |
| I12 | **Risk Budget Monitoring** — limit registration for tracking_error, var_95_pct_aum, max_drawdown, beta_max; utilisation % with ok/warning/breached status; NATS breach events | Risk/Governance | MSCI RiskMetrics, BlackRock Aladdin |
| I13 | **Portfolio Scoring and Rating Engine** — composite 0-100 score across Sharpe, concentration, ESG, compliance, and fee efficiency; letter grade (A+ to D); trend delta | Analytics | Morningstar Portfolio Rating, Fitch |
| I14 | **DRIP Automation** — per-instrument reinvestment policies (configurable %), automatic `add_holding` on dividend corporate actions, residual cash recording, fractional unit support | Operations | Computershare DRIP, FNZ Platform |
| I15 | **NATS-Backed Persistent Audit Log with Event Sourcing** — JetStream durable publish on every audit event; `replay_audit_events` for point-in-time portfolio reconstruction; tamper-evident | Infrastructure | Bytewax Event Sourcing, AWS EventBridge |

## New Methods

### 1. Factor Risk Decomposition

```python
# First record factor loadings per instrument
await svc.record_factor_loadings("SCOM.NSE", {
    "equity_beta": 1.15,
    "duration_dv01": 0.0,
    "credit_spread_dv01": 0.0,
    "fx_delta": 0.3,
    "real_estate_beta": 0.0,
})

result = await svc.factor_risk_decomposition("pf-001")
# {
#   "portfolio_volatility": 0.231,
#   "diversification_ratio": 1.0,
#   "factor_exposures": {"equity_beta": 0.892, "fx_delta": 0.142, ...},
#   "mctr": [{"instrument_id": "SCOM.NSE", "weight": 0.65, "mctr": {...}}, ...]
# }
```

### 2. Pre-Trade Compliance Check

```python
result = await svc.pre_trade_compliance_check("pf-001", proposed_trades=[
    {"asset_id": "EQTY.NSE", "action": "buy", "quantity_minor": 500_000},
    {"asset_id": "SANCTIONED-CO", "action": "buy", "quantity_minor": 100_000},
])
# {
#   "all_passed": False,
#   "total_violations": 1,
#   "trade_results": [
#     {"asset_id": "EQTY.NSE", "passed": True, "violations": []},
#     {"asset_id": "SANCTIONED-CO", "passed": False,
#      "violations": ["prohibited_instrument:sanctions_list"]},
#   ]
# }
```

### 3. Risk Budget Monitoring

```python
await svc.register_risk_budget("budget-01", "pf-001", limits={
    "var_95_pct_aum": 0.05,       # 5% VaR limit
    "max_drawdown": -0.15,         # 15% max drawdown
    "tracking_error": 0.04,        # 4% TE vs benchmark
    "beta_max": 1.2,
}, warning_pct=0.80)

status = await svc.monitor_risk_budget("pf-001")
# {
#   "any_breach": False,
#   "metrics": [
#     {"metric": "var_95_pct_aum", "current": 0.038, "limit": 0.05,
#      "utilisation_pct": 76.0, "status": "warning"},
#     ...
#   ]
# }
```

### 4. Household Consolidated View

```python
await svc.create_household("hh-001", client_id="client-42",
                            portfolio_ids=["pf-001", "pf-002", "pf-003"])

view = await svc.consolidated_portfolio_view(["pf-001", "pf-002", "pf-003"])
# {
#   "total_aum_minor": 45_000_000,
#   "herfindahl_index": 0.08,
#   "concentration_label": "low",
#   "blended_esg_composite": 68.4,
#   "allocation_breakdown": {"SCOM.NSE": 0.31, "KCB.NSE": 0.22, ...},
#   "portfolios": [...]
# }
```

### 5. DRIP Automation

```python
# Register policy: reinvest 100% of dividends for this instrument
await svc.register_drip_policy("pf-001", "SCOM.NSE",
                                reinvestment_pct=1.0, fractional_allowed=True)

# When a dividend corporate action fires, DRIP executes automatically
await svc.record_corporate_action(
    action_id="ca-055", instrument_id="SCOM.NSE",
    action_type="dividend", effective_date="2025-12-15",
    evidence_reference="DIV-NOTICE-2025-Q4",
    amount_minor=25_000, market_price_minor=185,
)
# Units reinvested = floor(25000 / 185) = 135 units added to pf-001 automatically
```

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
| prohibited_instrument_blocked | Trade against prohibited instrument list | deny |
| concentration_limit_enforced | Post-trade single-issuer weight exceeds 10% AUM | deny |
| glide_path_allocation_totals_100 | Glide path waypoint allocation does not sum to 100% | deny |
| risk_budget_breach_notified | Risk metric exceeds registered budget limit | notify+breach |
| drip_reinvestment_requires_market_price | DRIP process called with zero market price | deny |

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
| TaxLot | lot_id, holding_id, portfolio_id, instrument_id, purchase_date, quantity, cost_per_unit_minor, currency |
| GlidePath | id, tenant_id, portfolio_id, target_date, waypoints |
| DRIPPolicy | id, tenant_id, portfolio_id, instrument_id, reinvestment_pct, fractional_allowed |

## Streaming Events

Events emitted to the fintech event stream via Bytewax and NATS.

| Event | Trigger | Subject |
|-------|---------|---------|
| portfolio_book_created | Portfolio book created | apg.fintech.portfolio.lifecycle |
| portfolio_holding_recorded | Holding recorded | apg.fintech.portfolio.lifecycle |
| allocation_policy_activated | Allocation policy activated | apg.fintech.portfolio.lifecycle |
| portfolio_valuation_recorded | Valuation recorded | apg.fintech.portfolio.lifecycle |
| benchmark_assigned | Benchmark assigned | apg.fintech.portfolio.lifecycle |
| risk_exposure_recorded | Risk exposure recorded | apg.fintech.portfolio.risk |
| performance_attribution_recorded | Attribution recorded | apg.fintech.portfolio.lifecycle |
| cash_movement_recorded | Cash movement recorded | apg.fintech.portfolio.lifecycle |
| corporate_action_recorded | Corporate action processed | apg.fintech.portfolio.lifecycle |
| compliance_breach_recorded | Breach recorded | apg.fintech.portfolio.compliance |
| portfolio_review_recorded | Review completed | apg.fintech.portfolio.governance |
| portfolio_agent_registered | AI agent registered | apg.fintech.portfolio.agents |
| risk_budget_breach | Risk metric exceeds registered limit | apg.fintech.portfolio.risk_budget_breach |
| drip_executed | DRIP reinvestment completed | apg.fintech.portfolio.lifecycle |
| glide_path_applied | Glide path waypoint applied | apg.fintech.portfolio.lifecycle |
| prohibited_instrument_registered | Prohibited instrument added | apg.fintech.portfolio.compliance |
| nav_tick | Real-time NAV update per held instrument | apg.fintech.portfolio.nav (NATS SSE) |
| audit_event | Every audit action (JetStream durable) | apg.fintech.portfolio.audit.{tenant_id} |

## Edge Cases Handled

- Allocation totals must equal exactly 100% — rounding errors (e.g., 99.99%) are not tolerated
- Valuations with zero market value are denied — forces explicit handling of empty portfolios
- Corporate actions apply to an instrument, not a portfolio — the same dividend affects holdings across portfolios
- Risk exposure as-of-date is required to prevent stale exposure records
- Holdings can have fractional quantities (ETF fractional shares) but cannot be zero or negative
- TWR requires at least two valuation records; fewer returns `insufficient_data`
- MWR (IRR) annualisation uses actual calendar distance to avoid compounding artifacts
- Stress test scenarios without a matching instrument_id fall back to `equity` then `default` shock keys
- Counterparty concentration only applies to holdings with an `issuer_id`; others group under `unattributed`
- ESG score aggregation skips unscored holdings rather than diluting the weighted average
- Portfolio cloning copies allocation policy but starts with zero holdings
- Liquidity scoring classifies holdings without ADV metadata as `locked` (worst case)
- Glide path waypoints must individually total 100% — rejected at registration, not at apply time
- Tax lot disposal falls back to aggregate average-cost when no lots have been explicitly recorded
- Pre-trade compliance auto-records `ComplianceBreach` for every blocked trade
- DRIP with zero market price returns an error dict rather than raising — prevents cascade during unavailable prices
- Risk budget `max_drawdown` uses absolute value comparison so negative limits (e.g. -0.15) work correctly
- Consolidated portfolio view skips portfolios outside the tenant's scope rather than raising

## Composability

- **Upstream**: `fintech_wealth` provides client profile and mandate context; `fintech_robo` provides model portfolio templates; market data feeds are adapter boundaries referenced by ID
- **Downstream**: `fintech_trading` consumes portfolio positions for order generation; `bia` and `fin_rpt` consume valuations, attribution, and risk data for reporting
- **Peer**: Deployed alongside `fintech_wealth` (client-facing advisory layer) and `fintech_trading` (execution layer)

## Development Notes

- `treasury` portfolio type is included alongside standard investment types — supports corporate treasury management alongside client investment books
- Performance attribution `contributions` is a free-form dict; attribution methodology validation is the responsibility of the analytics adapter
- `market_data` is declared as an adapter in `DEFAULT_CONFIGURATION` but not in `REQUIRES` — soft dependency accessed via adapter reference at runtime
- `nats_url` is optional in `PortfolioManagementService.__init__`; without it, NATS events are silently skipped and the audit log remains in-memory only
- The `fintech_robo` dependency links robo advisory model portfolios to discretionary/model portfolio books, enabling automated rebalancing signals
