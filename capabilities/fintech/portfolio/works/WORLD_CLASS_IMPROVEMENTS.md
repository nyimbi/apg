# Portfolio Management — World-Class Improvement Roadmap

**Capability**: `fintech_portfolio` | **Version target**: 2.0.0

---

## 1. Multi-Currency FX Revaluation Engine

Current valuation uses a single `base_currency` per portfolio. Portfolios with cross-currency holdings (e.g. USD equities in a KES portfolio) use cost-basis minor units without FX conversion, producing NAV overstatement or understatement whenever the KES/USD rate moves.

**Improvement**: Add an FX rate store (`fx_rates: dict[str, dict[str, float]]`) keyed by `(date, pair)`. Expose `record_fx_rate(base, quote, rate, as_of_date)` and modify `portfolio_valuation` to apply live FX conversion per holding currency before summing to portfolio NAV. Return a per-currency breakdown in every valuation record.

---

## 2. Time-Weighted Return (TWR) Calculation

`total_return_calculation` uses a simple start-to-end ratio that is distorted by external cash flows. GIPS-compliant performance measurement requires chain-linked TWR (sub-period returns between each cash flow date).

**Improvement**: Implement `time_weighted_return(portfolio_id, start_date, end_date)` using the Modified Dietz or daily valuation chain-link method. Store TWR as a first-class performance metric alongside `PerformanceAttribution`.

---

## 3. Money-Weighted Return (MWR / IRR)

Institutional clients and private equity funds require Internal Rate of Return (IRR) to evaluate manager skill net of client cash flows.

**Improvement**: Add `money_weighted_return(portfolio_id, start_date, end_date)` using Newton-Raphson iteration on the NPV equation over all recorded `CashMovement` records and the ending NAV. Return annualised IRR, MOIC (multiple on invested capital), and DPI (distributions to paid-in).

---

## 4. Scenario Analysis / Stress Testing

`risk_metrics` computes parametric VaR under a single-scenario normal assumption. Tail risk events (2008, 2020 COVID, 2022 rate shock) are non-normal and require multi-scenario stress testing.

**Improvement**: Add `stress_test(portfolio_id, scenarios: list[dict])` where each scenario specifies per-asset-class shock factors. Compute shocked NAV, shocked VaR, and estimated drawdown per scenario. Return scenario ranking by loss severity.

---

## 5. Factor Risk Decomposition (Barra-style)

Holdings-level risk is currently aggregated into a single Herfindahl index and portfolio-level beta. Multi-asset portfolios need factor exposure decomposition (equity beta, duration, credit spread DV01, FX delta) for institutional risk reporting.

**Improvement**: Add `factor_risk_decomposition(portfolio_id)` that maps each holding to a factor exposure vector. Aggregate to portfolio-level exposures and return marginal contribution to risk (MCTR) per asset, enabling attribution of portfolio risk to individual holdings.

---

## 6. Liquidity Risk Scoring

Portfolios holding illiquid instruments (private equity, unlisted bonds) face redemption risk that VaR does not capture. CMA Kenya rules require LCR-like metrics for collective investment schemes.

**Improvement**: Add `liquidity_risk_score(portfolio_id, horizon_days: int)` that estimates days-to-liquidate for each holding based on ADV (average daily volume) metadata. Return a portfolio-level liquidity score, % assets liquidatable within 1/5/10/30 days, and any positions exceeding 25% of ADV.

---

## 7. ESG Score Aggregation

Institutional mandates increasingly require ESG screening. Portfolio-level ESG scores enable mandate compliance reporting and client impact dashboards.

**Improvement**: Add `esg_portfolio_score(portfolio_id)` that accepts per-instrument ESG scores (E, S, G sub-scores) via `record_esg_rating(instrument_id, e_score, s_score, g_score, source)`. Aggregate to weighted portfolio ESG scores and flag holdings breaching exclusion criteria.

---

## 8. Cost Basis Lot Tracking (Specific Lot ID)

Current `tax_lot_accounting` infers lots from aggregate holding records. Proper capital gains tax optimisation (e.g. maximising long-term treatment) requires lot-level cost basis tracking with purchase date, quantity, and price.

**Improvement**: Add a `TaxLot` model with `(lot_id, holding_id, purchase_date, quantity, cost_per_unit, currency)`. Modify `add_holding` to create a new lot each time. Implement `dispose_lots(portfolio_id, asset_id, quantity, method)` to select lots by FIFO/LIFO/highest-cost and compute per-lot realised gains with holding period.

---

## 9. Real-Time NAV Streaming via WebSocket / SSE

`portfolio_valuation` is a pull-based request-response operation. High-frequency valuation consumers (dashboards, risk systems) require push-based NAV updates as prices tick.

**Improvement**: Add `stream_nav_updates(portfolio_id)` as an async generator yielding `NavTick` events when underlying price feeds update. Integrate with the existing Bytewax event stream (`apg.fintech.portfolio.lifecycle`) and expose as a Server-Sent Events endpoint in `api.py`.

---

## 10. Automated Rebalancing Execution Integration

`rebalance_portfolio` returns suggested trades but does not execute them. Production rebalancing workflows require atomic order routing to `fintech_trading` with pre-trade compliance checks, order grouping across portfolios (block trading), and post-trade reconciliation.

**Improvement**: Add `execute_rebalance(portfolio_id, approved_by: str)` that: (1) calls `validate_agent_action` for human approval, (2) generates order records, (3) emits order events to `apg.fintech.trading.orders`, and (4) records a pending-reconciliation state. Complete with `confirm_rebalance_execution(portfolio_id, execution_report)` to reconcile final fills.

---

## 11. Portfolio Clone / Template Instantiation

Model portfolio workflows require cloning a template portfolio with defined allocation weights to a new client book. Currently requires manually recreating every allocation policy and holding.

**Improvement**: Add `clone_portfolio(source_portfolio_id, target_client_id, name, override_allocations)` that deep-copies the allocation policy and optionally the holdings of a model/template portfolio into a new portfolio for a different client. Track `cloned_from` provenance in the new `PortfolioBook`.

---

## 12. Audit Trail Query & Export

`audit_events` is an in-memory list with no query interface. Production systems require auditor-accessible, immutable audit logs with search by event type, date range, actor, and reference ID.

**Improvement**: Add `query_audit_events(event_type, reference_id, start_dt, end_dt, limit)` returning paginated, time-ordered audit records. Add `export_audit_log(portfolio_id, fmt)` to produce CSV/JSON audit exports suitable for regulatory submission. Persist to a dedicated `audit_log` table via the DB adapter.

---

## 13. Counterparty Exposure Aggregation

Holdings across multiple portfolios can be concentrated in the same issuer (counterparty). CMA and Basel III rules impose single-counterparty limits across all portfolios under management.

**Improvement**: Add `counterparty_exposure_summary()` that scans all tenant portfolios and aggregates holdings by `issuer_id` (a new optional field on `HoldingRecord`). Return total exposure, % of total AUM, and a breach flag if any counterparty exceeds a configurable limit (default 10%).

---

## 14. Target Date / Glide Path Management

Target-date funds require automatic allocation glide paths that de-risk toward a target date by shifting from equity to fixed income over time.

**Improvement**: Add `GlidePath` model with `(portfolio_id, target_date, waypoints: list[AllocationPolicy])` where each waypoint specifies target allocations at a future date. Add `apply_glide_path(portfolio_id)` that selects the nearest future waypoint and rebalances toward it, recording the policy change with full audit trail.

---

## 15. Client-Facing Performance Report Generation

Currently `regulatory_reporting` generates compliance-oriented filings. Clients require branded, plain-language performance reports (IPS quarterly, annual review) with charts and attribution narrative.

**Improvement**: Add `generate_client_report(portfolio_id, period, template: str)` that assembles performance attribution, risk metrics, drawdown, income distribution, and benchmark comparison into a structured report payload. Template options: `ips_quarterly`, `annual_review`, `factsheet`. Return a report dict ready for PDF rendering via a document generation adapter.
