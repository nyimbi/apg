# Portfolio Management

**Capability ID**: `fintech_portfolio` | **Domain**: `fintech` | **Version**: `3.0.0`

## Description

Portfolio Management provides regulated investment book operations: portfolio book creation, holding ledger recording, allocation policy activation (totals must equal exactly 100%), valuation capture, benchmark assignment, risk exposure tracking, performance attribution, cash movement recording, corporate action processing, compliance breach recording, and governance reviews.

Version 3.0.0 builds on the v2.0.0 GIPS-compliant TWR/MWR, stress testing, counterparty risk, FX, cloning, ESG, and audit log capabilities with the following additions:

- **Factor risk decomposition** — Barra-style MCTR per holding and portfolio-level factor exposures
- **Liquidity risk scoring** — ADV-based days-to-liquidate classification and portfolio liquidity score
- **Glide path management** — Target-date fund waypoints with automatic de-risking rebalancing
- **Tax lot disposal** — Specific-lot FIFO/LIFO/highest-cost with Kenya Finance Act 2023 CGT at 15%
- **Pre-trade compliance** — Prohibited instrument list, concentration limits, and mandate checks
- **Risk budget monitoring** — Limit utilisation tracking with auto-breach recording
- **Transaction cost analysis** — Implementation shortfall, VWAP slippage, and broker ranking
- **Household consolidated view** — Multi-portfolio AUM aggregation with blended ESG and allocation
- **DRIP automation** — Dividend reinvestment with fractional unit support and residual cash recording

## Installation

```bash
pip install apg-fintech-portfolio
```

## Provides

- `portfolio_book_workflow`
- `portfolio_holding_workflow`
- `portfolio_allocation_policy_workflow`
- `portfolio_valuation_workflow`
- `portfolio_benchmark_workflow`
- `portfolio_risk_workflow`
- `portfolio_attribution_workflow`
- `portfolio_cash_workflow`
- `portfolio_corporate_action_workflow`
- `portfolio_compliance_workflow`
- `portfolio_review_workflow`
- `portfolio_agent_workflow`
- `portfolio_twr_workflow`
- `portfolio_mwr_workflow`
- `portfolio_stress_test_workflow`
- `portfolio_counterparty_workflow`
- `portfolio_fx_workflow`
- `portfolio_clone_workflow`
- `portfolio_audit_query_workflow`
- `portfolio_client_report_workflow`
- `portfolio_esg_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-portfolio/dashboard` | `fintech_portfolio:view` | Overview |
| `/fintech-portfolio/portfolios` | `fintech_portfolio:portfolios` | Books |
| `/fintech-portfolio/holdings` | `fintech_portfolio:holdings` | Books |
| `/fintech-portfolio/allocations` | `fintech_portfolio:allocations` | Policy |
| `/fintech-portfolio/valuations` | `fintech_portfolio:valuations` | Operations |
| `/fintech-portfolio/benchmarks` | `fintech_portfolio:benchmarks` | Policy |
| `/fintech-portfolio/risk` | `fintech_portfolio:risk` | Risk |
| `/fintech-portfolio/attribution` | `fintech_portfolio:attribution` | Performance |
| `/fintech-portfolio/twr` | `fintech_portfolio:performance` | Performance |
| `/fintech-portfolio/mwr` | `fintech_portfolio:performance` | Performance |
| `/fintech-portfolio/stress-test` | `fintech_portfolio:risk` | Risk |
| `/fintech-portfolio/counterparty-exposure` | `fintech_portfolio:risk` | Risk |
| `/fintech-portfolio/fx-rates` | `fintech_portfolio:operations` | Operations |
| `/fintech-portfolio/clone` | `fintech_portfolio:admin` | Administration |
| `/fintech-portfolio/audit` | `fintech_portfolio:admin` | Administration |
| `/fintech-portfolio/client-report` | `fintech_portfolio:view` | Reports |
| `/fintech-portfolio/esg` | `fintech_portfolio:view` | ESG |

## Key Service Methods

### Portfolio Lifecycle
- `create_portfolio(name, client_id, strategy, benchmark, portfolio_type, base_currency, policy_reference)`
- `get_portfolio(portfolio_id)`
- `list_portfolios(client_id, portfolio_type)`
- `close_portfolio(portfolio_id, reason)`
- `clone_portfolio(source_portfolio_id, target_client_id, name, override_allocations)`

### Holdings
- `add_holding(portfolio_id, asset_id, quantity, cost_basis, currency)`
- `remove_holding(portfolio_id, asset_id, quantity, proceeds)`
- `get_holding(portfolio_id, asset_id)`
- `list_holdings(portfolio_id)`
- `bulk_add_holdings(portfolio_id, holdings)`

### Valuation & Allocation
- `portfolio_valuation(portfolio_id, as_of_date, source_reference)`
- `activate_allocation_policy(allocation_id, portfolio_id, target_allocation, policy_reference)`
- `rebalance_portfolio(portfolio_id)`

### Performance
- `performance_attribution(portfolio_id, period, benchmark_id)` — Brinson-Hood-Beebower attribution
- `sharpe_ratio(portfolio_id, period)` — annualised Sharpe from valuation history
- `drawdown_analysis(portfolio_id)` — max drawdown and current drawdown with peak/trough dates
- `total_return_calculation(portfolio_id, period)` — capital gain + income return
- `time_weighted_return(portfolio_id, start_date, end_date)` — GIPS-compliant chain-linked TWR
- `money_weighted_return(portfolio_id, start_date, end_date)` — IRR, MOIC, DPI for closed-end funds
- `benchmark_tracking_error(portfolio_id, benchmark_id)` — annualised active return standard deviation

### Risk
- `risk_metrics(portfolio_id)` — VaR 95/99, CVaR, beta, Herfindahl concentration
- `record_risk_exposure(exposure_id, portfolio_id, metric, value, as_of_date, source_reference)`
- `stress_test(portfolio_id, scenarios)` — multi-scenario shocked NAV and drawdown
- `counterparty_exposure_summary()` — single-counterparty CMA concentration across all portfolios
- `factor_risk_decomposition(portfolio_id)` — Barra-style MCTR per holding; requires `record_factor_loadings`
- `liquidity_risk_score(portfolio_id, horizon_days, participation_rate)` — days-to-liquidate classification; requires `record_liquidity_metadata`
- `register_risk_budget(budget_id, portfolio_id, limits, warning_pct)` — register metric limits
- `monitor_risk_budget(portfolio_id)` — utilisation monitoring with auto-breach on limit breach

### ESG
- `record_esg_rating(instrument_id, e_score, s_score, g_score, source, excluded)`
- `esg_portfolio_score(portfolio_id)` — weighted E/S/G scores and exclusion breach detection

### FX & Currency
- `record_fx_rate(base_currency, quote_currency, rate, as_of_date, source_reference)` — store FX rates for multi-currency revaluation

### Reporting
- `regulatory_reporting(portfolio_id, report_type)` — UCITS, AIFMD, MiFID_TRANSACTION, SORP, IPS_QUARTERLY, CMA_QUARTERLY
- `generate_client_report(portfolio_id, period, template)` — structured client report (ips_quarterly, annual_review, factsheet)
- `cma_portfolio_return(period)` — CMA Kenya investment manager return
- `income_distribution_report(portfolio_id, period)`
- `cash_flow_projection(portfolio_id, months)`
- `export_portfolio_data(portfolio_id, fmt)`
- `consolidated_portfolio_view(portfolio_ids)` — household/sleeve AUM, allocation, and blended ESG
- `transaction_cost_analysis(portfolio_id, period)` — shortfall, broker ranking; requires `record_trade_execution`

### Compliance & Operations
- `record_compliance_breach(breach_id, portfolio_id, severity, evidence_reference)`
- `record_review(review_id, reference_id, reviewer_id, status, evidence_reference)`
- `position_reconciliation(portfolio_id, custodian_report)` — matched/break/internal-only/custodian-only
- `record_cash_movement(movement_id, portfolio_id, amount_minor, currency, reference)`
- `record_corporate_action(action_id, instrument_id, action_type, effective_date, evidence_reference, ratio)`
- `pre_trade_compliance_check(portfolio_id, proposed_trades)` — prohibited list + concentration + mandate
- `register_prohibited_instrument(instrument_id, reason, effective_date)`

### Tax & Lots
- `dispose_lots(portfolio_id, asset_id, quantity, method, proceeds_minor)` — FIFO/LIFO/highest-cost with Kenya CGT

### Glide Path
- `register_glide_path(portfolio_id, target_date, waypoints)`
- `apply_glide_path(portfolio_id)` — apply nearest future waypoint
- `glide_path_schedule(portfolio_id)` — full schedule with days-to-waypoint

### DRIP
- `register_drip_policy(portfolio_id, instrument_id, reinvestment_pct, fractional_allowed)`
- `process_drip(portfolio_id, instrument_id, dividend_per_unit_minor, market_price_minor)`

### Household
- `create_household(household_id, client_id, portfolio_ids, name)`
- `consolidated_portfolio_view(portfolio_ids)`

### Audit
- `query_audit_events(event_type, reference_id, start_dt, end_dt, limit)` — filterable, paginated audit log
- `dashboard_summary()` — aggregate tenant-level counts

### Agents & Batch
- `register_portfolio_agent(agent_id, name, runtime, role, scope)`
- `validate_agent_action(privileged_scope, human_approval_recorded)`
- `validate_batch(item_count, event_stream)`

_(See `service.py` for complete signatures and inline docstrings.)_

## Usage Examples

### Create a portfolio and add holdings

```python
from capabilities.fintech.portfolio.service import PortfolioManagementService

svc = PortfolioManagementService(tenant_id="datacraft")

# Create portfolio
portfolio = await svc.create_portfolio(
    name="Growth Fund A",
    client_id="client-001",
    strategy="equity_growth",
    benchmark="NSE20",
    portfolio_type="discretionary",
    base_currency="KES",
    policy_reference="IPS-2025-001",
)

# Add holdings
await svc.add_holding(portfolio["portfolio_id"], "SCOM", 10000, 42.5, "KES")
await svc.add_holding(portfolio["portfolio_id"], "KCB", 5000, 38.0, "KES")
```

### GIPS-compliant performance reporting

```python
# Record daily valuations over time
await svc.portfolio_valuation(pid, "2025-01-31", "pricing_service")
await svc.portfolio_valuation(pid, "2025-02-28", "pricing_service")
await svc.portfolio_valuation(pid, "2025-03-31", "pricing_service")

# Chain-linked time-weighted return (TWR)
twr = await svc.time_weighted_return(pid, "2025-01-01", "2025-03-31")
print(twr["annualised_twr"])   # e.g. 0.142

# Money-weighted return (IRR) for closed-end fund reporting
mwr = await svc.money_weighted_return(pid, "2025-01-01", "2025-03-31")
print(mwr["moic"], mwr["irr_annualised"])
```

### Stress testing

```python
scenarios = [
    {"name": "2020_covid", "shocks": {"equity": -0.35, "default": -0.20}},
    {"name": "rate_shock_300bps", "shocks": {"fixed_income": -0.15, "default": -0.05}},
    {"name": "fx_depreciation_30pct", "shocks": {"SCOM": -0.10, "default": -0.02}},
]
result = await svc.stress_test(pid, scenarios)
print(result["worst_case_scenario"])
```

### ESG scoring

```python
# Load instrument-level ESG ratings
await svc.record_esg_rating("SCOM", e_score=72, s_score=65, g_score=80, source="MSCI_ESG")
await svc.record_esg_rating("ARMS", e_score=15, s_score=30, g_score=55, source="MSCI_ESG", excluded=True)

# Get weighted portfolio ESG score
esg = await svc.esg_portfolio_score(pid)
print(esg["composite_score"])         # 0–100
print(esg["exclusion_breaches"])      # ["ARMS"]
```

### Clone a model portfolio

```python
# Clone model portfolio to a new client, overriding allocations
new = await svc.clone_portfolio(
    source_portfolio_id=model_pid,
    target_client_id="client-042",
    name="Client 042 Growth",
    override_allocations={"SCOM": 0.40, "KCB": 0.30, "EQTY": 0.30},
)
print(new["portfolio"]["portfolio_id"])
```

### Generate a client factsheet

```python
report = await svc.generate_client_report(pid, period="Q1_2025", template="factsheet")
# report contains performance, risk, drawdown, income, benchmarks, holdings
```

### Query the audit log

```python
events = await svc.query_audit_events(
    event_type="compliance_breach_recorded",
    start_dt="2025-01-01T00:00:00",
    limit=50,
)
print(events["total_matched"])
```

### Factor risk decomposition

```python
# Store factor loadings for each instrument
await svc.record_factor_loadings("SCOM", equity_beta=0.85, fx_delta=0.0)
await svc.record_factor_loadings("KCB", equity_beta=1.10, duration_dv01=0.0)
await svc.record_factor_loadings("KE20Y", equity_beta=0.05, duration_dv01=8.5)

result = await svc.factor_risk_decomposition(pid)
print(result["factor_exposures"])   # {"equity_beta": 0.87, "duration_dv01": 1.2, ...}
print(result["portfolio_volatility"])
for h in result["mctr"]:
    print(h["instrument_id"], h["mctr"]["equity_beta"])
```

### Liquidity risk scoring

```python
# Load ADV and spread data for each instrument
await svc.record_liquidity_metadata("SCOM", adv_minor=5_000_000_00, bid_ask_spread_bps=12)
await svc.record_liquidity_metadata("KCB",  adv_minor=3_000_000_00, bid_ask_spread_bps=18)

liq = await svc.liquidity_risk_score(pid, horizon_days=10, participation_rate=0.20)
print(liq["liquidity_score"])       # 0–100
print(liq["bucket_pcts"])           # {"liquid": 72.4, "semi_liquid": 20.1, ...}
print(liq["holdings_over_25pct_adv"])  # positions that dominate ADV
```

### Glide path management (target-date funds)

```python
waypoints = [
    {"date": "2027-01-01", "allocation": {"equity": 0.70, "fixed_income": 0.25, "cash": 0.05}},
    {"date": "2029-01-01", "allocation": {"equity": 0.50, "fixed_income": 0.42, "cash": 0.08}},
    {"date": "2031-01-01", "allocation": {"equity": 0.30, "fixed_income": 0.60, "cash": 0.10}},
]
await svc.register_glide_path(pid, target_date="2031-12-31", waypoints=waypoints)

# Apply the nearest future waypoint — triggers an allocation policy activation
result = await svc.apply_glide_path(pid)
print(result["applied_waypoint_date"], result["remaining_waypoints"])

# See the full schedule
schedule = await svc.glide_path_schedule(pid)
for wp in schedule["schedule"]:
    print(wp["date"], wp["days_to_waypoint"], wp["status"])
```

### Tax lot disposal with Kenya CGT

```python
# dispose_lots falls back gracefully if lots have not been explicitly recorded
result = await svc.dispose_lots(
    portfolio_id=pid,
    asset_id="SCOM",
    quantity=2000,
    method="fifo",               # fifo | lifo | highest_cost
    proceeds_minor=900_000_00,   # KES 900,000 in minor units
)
print(result["total_gain_minor"])
print(result["cgt_liability_minor"])  # @ 15% Finance Act 2023
for lot in result["lots_disposed"]:
    print(lot["purchase_date"], lot["gain_minor"], lot["holding_days"])
```

### Pre-trade compliance checking

```python
# Register sanctions / exclusion list
await svc.register_prohibited_instrument("XYZ_CORP", reason="sanctions_list", effective_date="2026-01-01")

# Check proposed trades before routing to execution
check = await svc.pre_trade_compliance_check(
    pid,
    proposed_trades=[
        {"asset_id": "SCOM",     "action": "buy",  "quantity_minor": 50_000_00},
        {"asset_id": "XYZ_CORP", "action": "buy",  "quantity_minor": 10_000_00},
    ],
)
print(check["all_passed"])          # False — XYZ_CORP blocked
for t in check["trade_results"]:
    print(t["asset_id"], t["passed"], t["violations"])
```

### Risk budget monitoring

```python
await svc.register_risk_budget(
    "budget-001", pid,
    limits={
        "var_95_pct_aum": 0.05,     # max 5% daily VaR
        "max_drawdown":   -0.15,    # max 15% drawdown
        "beta_max":       1.20,     # max portfolio beta
        "tracking_error": 0.06,     # max 6% TE vs benchmark
    },
    warning_pct=0.80,
)

status = await svc.monitor_risk_budget(pid)
print(status["any_breach"])
for m in status["metrics"]:
    print(m["metric"], f"{m['utilisation_pct']}%", m["status"])
```

### Transaction cost analysis

```python
# Record fills after execution
await svc.record_trade_execution(
    "trade-001", pid, "SCOM",
    decision_price_minor=4250,    # price at decision time
    execution_price_minor=4267,   # actual fill
    quantity=10_000,
    broker_id="broker-stanbic",
    execution_time="2026-06-01T09:35:00Z",
)

tca = await svc.transaction_cost_analysis(pid, period="Q2_2026")
print(tca["aggregate_shortfall_bps"])      # bps of AUM
print(tca["broker_ranking"])               # sorted by avg shortfall
```

### Household consolidated view

```python
household = await svc.create_household(
    "hh-001", "client-007",
    portfolio_ids=[pension_pid, discretionary_pid, isa_pid],
    name="Kamau Family Office",
)

view = await svc.consolidated_portfolio_view([pension_pid, discretionary_pid, isa_pid])
print(view["total_aum_minor"])
print(view["blended_esg_composite"])
print(view["allocation_breakdown"])   # {instrument_id: weight}
```

### DRIP automation

```python
# Register reinvestment policy (100% reinvest, fractional units OK)
await svc.register_drip_policy(pid, "SCOM", reinvestment_pct=1.0, fractional_allowed=True)

# Triggered automatically when record_corporate_action is called for a dividend,
# or call directly with the dividend rate and current market price:
drip_result = await svc.process_drip(
    pid, "SCOM",
    dividend_per_unit_minor=150,   # KES 1.50 per share in minor units
    market_price_minor=4300,       # current market price
)
print(drip_result["units_reinvested"])
print(drip_result["residual_cash_minor"])  # fractional share residual as cash
```

## Interoperability

`fintech_portfolio` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_portfolio;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_PORTFOLIO_`.

## Further Reading

- `service.py` — Business logic implementation (all async methods)
- `models.py` — Data models (Pydantic v2)
- `api.py` — REST API endpoints (Flask-AppBuilder blueprints)
- `views.py` — Flask-AppBuilder views and Pydantic request/response schemas
- `capability_contract.py` — Rule engine contract and supported constant sets
- `README.md` — Quick reference, business rules, streaming events
- `works/WORLD_CLASS_IMPROVEMENTS.md` — Improvement roadmap with 15 detailed enhancements
