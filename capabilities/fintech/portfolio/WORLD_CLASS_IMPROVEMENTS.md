# Portfolio Management — World-Class Improvement Roadmap

**Capability**: `fintech_portfolio` | **Version target**: 3.0.0 | **Date**: 2026-06

---

### I1. Factor Risk Decomposition (Barra-style MCTR)

**Category**: Risk Analytics
**Justification**: The current `risk_metrics` collapses all portfolio risk into a scalar VaR and a single portfolio beta. Multi-asset portfolios (equities + bonds + real estate + alternatives) require factor exposure decomposition to understand _where_ risk is coming from. Without it, the risk team cannot perform limit attribution or isolate the source of a VaR breach. This is table-stakes for any institutional asset manager — Bloomberg PORT, MSCI RiskMetrics, and Northfield all provide it.
**Implementation**: Add `factor_risk_decomposition(portfolio_id)`. Map each holding to a factor vector `{equity_beta, duration_dv01, credit_spread_dv01, fx_delta, real_estate_beta}` stored via `record_factor_loadings(instrument_id, factor_loadings)`. Aggregate to portfolio-level factor exposures using holding weights. Compute Marginal Contribution to Risk (MCTR) per holding: `MCTR_i = w_i * (Sigma * w)_i / sigma_p`. Return per-factor attribution, per-holding MCTR, and diversification ratio.
**Competitor**: MSCI RiskMetrics, Bloomberg PORT, Axioma Qontigo

---

### I2. Liquidity Risk Scoring with Days-to-Liquidate

**Category**: Risk Analytics
**Justification**: VaR assumes positions can be exited at the mark price. For thinly traded NSE-listed stocks, corporate bonds, or private equity stakes, the bid-ask spread and ADV (average daily volume) govern _actual_ liquidation cost. CMA Kenya's CIS regulations require LCR-like disclosure for open-ended funds. Ignoring liquidity risk produces dangerously optimistic VaR estimates and can trigger forced liquidation cascades.
**Implementation**: Add `liquidity_risk_score(portfolio_id, horizon_days: int = 10)`. Require `record_liquidity_metadata(instrument_id, adv_minor, bid_ask_spread_bps, market_cap_minor)`. Estimate days-to-liquidate per holding as `ceil(quantity * price / (participation_rate * adv))`. Classify holdings as liquid (≤1 day), semi-liquid (2–5 days), illiquid (6–30 days), locked (>30 days). Return portfolio liquidity score (0–100), % AUM liquidatable within 1/5/10/30 days, and any position exceeding 25% of ADV.
**Competitor**: BlackRock Aladdin Liquidity Stress Testing, MSCI Liquidity Risk

---

### I3. Automated Rebalancing Execution with Block Trading

**Category**: Operations / Execution
**Justification**: `rebalance_portfolio` produces suggested trades but stops short of execution. Portfolio managers using the system must manually enter orders, introducing delay, fat-finger risk, and compliance gaps. Institutional PMS systems (Charles River, Advent Geneva, SimCorp Dimension) execute rebalancing atomically with pre-trade compliance checks and multi-portfolio block trading to minimise market impact.
**Implementation**: Add `execute_rebalance(portfolio_id, approved_by: str)` that: (1) calls `validate_agent_action` for human approval gate, (2) emits order intents to `apg.fintech.trading.orders` via NATS subject with JetStream durable consumer, (3) records a `PENDING_EXECUTION` state per suggested trade, (4) locks the portfolio allocation policy until all orders settle. Add `confirm_rebalance_execution(portfolio_id, execution_report: list[dict])` to reconcile final fills, update holdings, and release the lock. Use NATS for the order event bus — not Kafka — consistent with APG streaming architecture.
**Competitor**: Charles River IMS, SimCorp Dimension, Advent Geneva

---

### I4. Target-Date Glide Path Management

**Category**: Product / Strategy
**Justification**: Target-date funds (TDFs) are the fastest-growing fund category in East Africa's NSSF and pension space. They require automatic de-risking as the target date approaches — shifting from equity to fixed income along a predefined glide path. Without glide path management, a TDF is indistinguishable from a static balanced fund and cannot be sold as target-date compliant.
**Implementation**: Add `GlidePath` model: `(id, tenant_id, portfolio_id, target_date: str, waypoints: list[dict])` where each waypoint is `{date: str, allocation: dict[str, float]}`. Add `register_glide_path(portfolio_id, target_date, waypoints)`, `apply_glide_path(portfolio_id)` — selects the nearest future waypoint, calls `activate_allocation_policy`, and records a `glide_path_applied` audit event. Add `glide_path_schedule(portfolio_id)` returning the full waypoint table with days-to-next and de-risking velocity.
**Competitor**: Vanguard Target Retirement, Fidelity Freedom, BlackRock LifePath

---

### I5. Tax-Lot Tracking with Specific Identification

**Category**: Tax & Compliance
**Justification**: The current `tax_lot_accounting` uses aggregate holding records and cannot compute accurate capital gains taxes. In Kenya, the Finance Act 2023 introduced CGT on listed securities at 15%. Investors holding the same stock acquired at different prices and dates need specific-lot disposal to: (a) maximise long-term treatment, (b) harvest losses, (c) produce accurate tax certificates. Without lot-level tracking, the system exposes clients to CGT liability overstatement or understatement — both regulatory and fiduciary failures.
**Implementation**: Add `TaxLot` dataclass: `(lot_id, holding_id, portfolio_id, instrument_id, purchase_date, quantity, cost_per_unit_minor, currency)`. Modify `add_holding` to create a `TaxLot` entry in `self.tax_lots` dict alongside the aggregate holding update. Add `dispose_lots(portfolio_id, asset_id, quantity, method: str = "fifo")` supporting FIFO/LIFO/highest-cost/specific-lot selection. Return per-lot realised gain, holding period days, short/long-term classification, and CGT liability estimate.
**Competitor**: Advent Geneva Lot Tracker, SS&C Eze Eclipse, Interactive Brokers Tax Optimizer

---

### I6. Real-Time NAV Streaming via NATS + Bytewax

**Category**: Infrastructure / Streaming
**Justification**: `portfolio_valuation` is synchronous pull. Downstream consumers — risk dashboards, collateral engines, margin call systems — need push-based NAV updates with sub-second latency. A static snapshot every end-of-day is insufficient for intraday margin management or live client-facing apps. Bloomberg and Reuters provide real-time NAV for mutual funds; institutional risk platforms require the same.
**Implementation**: Add `async def stream_nav_updates(portfolio_id: str) -> AsyncGenerator[dict, None]` as an async generator. Subscribe to `apg.market_data.prices.>` on NATS (via `nats.py` async client). On each price tick for a held instrument, recompute the holding's marked value using last recorded FX rates, emit a `NavTick` dict `{portfolio_id, instrument_id, nav_minor, delta_minor, as_of}`. Expose as a Server-Sent Events endpoint at `/fintech-portfolio/{id}/nav-stream` in `api.py`. Integrate with Bytewax for downstream aggregation and windowed risk recomputation.
**Competitor**: Bloomberg B-PIPE Real-Time NAV, Refinitiv Eikon NAV feed

---

### I7. Multi-Portfolio Consolidated View (Household / Sleeve Management)

**Category**: Client UX / Analytics
**Justification**: Wealth management clients typically have multiple portfolios: pension fund, ISA wrapper, offshore trust, and a discretionary account. Advisors need a consolidated view of total AUM, consolidated asset allocation, and blended performance across all portfolios — a `household` or `sleeve` aggregation. Without it, advisors must manually sum portfolio reports and cannot detect cross-portfolio allocation drift or wash-sale violations.
**Implementation**: Add `consolidated_portfolio_view(portfolio_ids: list[str])` that aggregates total AUM, weighted asset allocation breakdown, blended TWR (money-weighted across the group), blended ESG score, combined counterparty concentration, and a drift-from-target table. Add `create_household(household_id, client_id, portfolio_ids)` stored in `self.households` dict with membership management methods `add_portfolio_to_household` and `remove_portfolio_from_household`.
**Competitor**: Orion Portfolio Solutions, Envestnet ENV2, Addepar

---

### I8. Pre-Trade Compliance Checking

**Category**: Compliance / Risk Control
**Justification**: Rebalancing proposals and manual orders must pass pre-trade compliance before hitting execution. CMA Kenya requires pre-trade screening for: (a) concentration limits (single issuer >10% AUM), (b) prohibited securities (politically exposed, sanctioned), (c) mandate drift (buying equity when mandate is fixed income). Without automated pre-trade checks, compliance officers become manual bottlenecks and regulators can impose penalties for guideline violations.
**Implementation**: Add `pre_trade_compliance_check(portfolio_id, proposed_trades: list[dict])` where each trade is `{asset_id, action, quantity_minor}`. Run checks: concentration limit post-trade, prohibited instrument list (stored via `register_prohibited_instrument`), mandate type alignment, and leverage limit. Return per-trade `{passed: bool, violations: list[str]}` and an aggregate `{all_passed: bool, total_violations: int}`. Record a `ComplianceBreach` for any failing trade with severity `high`. Integrate as a required gate in `execute_rebalance`.
**Competitor**: Charles River Compliance, Bloomberg AIM Pre-Trade, Fidessa Compliance Manager

---

### I9. Transaction Cost Analysis (TCA)

**Category**: Performance / Execution Quality
**Justification**: Understanding execution quality is mandatory for MiFID II best-execution obligations and is increasingly expected by institutional clients in Africa. Without TCA, portfolio managers cannot know whether broker selection, order timing, and trade size are eroding alpha. Implementation shortfall — the gap between decision price and execution price — is the gold standard metric, used by all bulge-bracket asset managers.
**Implementation**: Add `record_trade_execution(trade_id, portfolio_id, asset_id, decision_price_minor, execution_price_minor, quantity, broker_id, execution_time)`. Add `transaction_cost_analysis(portfolio_id, period: str)` computing: implementation shortfall per trade, VWAP slippage (vs recorded ADV metadata), market impact (Almgren-Chriss approximation using ADV and volatility), broker performance ranking, and aggregate TCA cost as bps of AUM. Return a detailed breakdown suitable for quarterly best-execution report to the board.
**Competitor**: ITG/Virtu TCA, Bloomberg TCA, Liquidnet Analytics

---

### I10. Attribution by Asset Class / Sector / Geography

**Category**: Performance Analytics
**Justification**: The existing `performance_attribution` implements Brinson-Hood-Beebower but only at portfolio level. Institutional reporting requires multi-level attribution drill-down: by asset class (equity vs bond vs cash), by sector (financials, technology, energy), by geography (Kenya, pan-Africa, global). Without this granularity, the investment committee cannot identify whether performance comes from country selection, sector rotation, or stock picking.
**Implementation**: Add `multi_level_attribution(portfolio_id, period, levels: list[str] = ["asset_class", "sector", "geography"])`. Require `record_instrument_classification(instrument_id, asset_class, sector, geography)` to populate a `self._instrument_classifications` store. For each level, compute BHB allocation effect, selection effect, and interaction effect by grouping holdings. Return a nested attribution tree with contribution at each level and a waterfall decomposition.
**Competitor**: FactSet PA, Bloomberg PORT Multi-Level Attribution, Morningstar Direct

---

### I11. Custom Benchmark Construction

**Category**: Performance / Analytics
**Justification**: Most portfolios are measured against composite benchmarks (e.g. 70% NSE20 + 30% FTSE EPRA Africa REIT). The current `BenchmarkAssignment` only stores an index ID. Composite benchmarks require: (a) storing constituent weights, (b) computing weighted benchmark return from constituent returns, (c) rebalancing the benchmark at the same frequency as the portfolio. Without composite benchmark support, active return calculations are approximate and misleading.
**Implementation**: Add `create_composite_benchmark(benchmark_id, name, constituents: list[dict])` where each constituent is `{index_id, weight}`. Store in `self.composite_benchmarks`. Add `record_index_return(index_id, period, return_pct)` to feed benchmark returns. Modify `performance_attribution` to fetch the composite benchmark return and compute genuine active return instead of the synthetic seed-based simulation. Add `benchmark_return_decomposition(portfolio_id, period)` showing the contribution of each constituent to total benchmark return.
**Competitor**: MSCI Custom Index Builder, FTSE Russell Index Builder, Bloomberg BCBM

---

### I12. Risk Budget Monitoring

**Category**: Risk / Governance
**Justification**: Institutional mandates define risk budgets — maximum active risk (tracking error), maximum VaR, maximum drawdown — that constrain portfolio construction. Currently the system records risk exposures but does not compare them against mandate limits or alert when limits are approached (warning) or breached (hard stop). Without risk budget monitoring, breaches are discovered after the fact in the compliance review cycle.
**Implementation**: Add `register_risk_budget(budget_id, portfolio_id, limits: dict[str, float], warning_pct: float = 0.80)` storing budgets in `self.risk_budgets`. The `limits` dict maps metric names (`tracking_error`, `var_95_pct_aum`, `max_drawdown`, `beta_max`) to numeric limits. Add `monitor_risk_budget(portfolio_id)` that computes current values for each limit metric (re-using existing risk methods), compares to budget, and returns `{metric, current, limit, utilisation_pct, status: ok|warning|breached}` per metric. Trigger a `ComplianceBreach` with severity `high` on breaches and `medium` on warnings. Emit a NATS event `apg.fintech.portfolio.risk_budget_breach`.
**Competitor**: MSCI RiskMetrics Risk Budget, BlackRock Aladdin Risk Monitor, Axioma Portfolio Optimizer

---

### I13. Portfolio Scoring & Rating Engine

**Category**: Analytics / Client Reporting
**Justification**: Retail investors and financial advisors need a simple portfolio health score (like a credit rating) that synthesises risk-adjusted performance, diversification, cost efficiency, ESG compliance, and mandate adherence into an actionable grade. Morningstar Star Ratings and Fitch Portfolio Ratings have mass-market recognition precisely because complexity collapses to a single number that drives AUM flow decisions.
**Implementation**: Add `compute_portfolio_score(portfolio_id)`. Pull: Sharpe ratio (normalised 0–25), Herfindahl concentration (inverted, 0–20), ESG composite (normalised 0–20), compliance breach count (inverted, 0–20), fee-to-return ratio (inverted, 0–15). Sum to composite score 0–100. Map to letter grade (A+ ≥ 90, A ≥ 80, B+ ≥ 70, B ≥ 60, C ≥ 50, D < 50). Return score, grade, per-dimension breakdown, and delta vs prior period. Store in `self.portfolio_scores` for trend tracking.
**Competitor**: Morningstar Portfolio Rating, Fitch Portfolio Assessment, MSCI Portfolio Score

---

### I14. Dividend Reinvestment Plan (DRIP) Automation

**Category**: Operations / Corporate Actions
**Justification**: Dividend reinvestment is the most common recurring portfolio operation — for most retail-oriented CIS schemes in Kenya, 100% of dividends are reinvested. Currently, dividends are recorded as cash movements and the reinvestment is a separate manual `add_holding` call. Without DRIP automation, the operational burden scales linearly with AUM and the window between ex-date and reinvestment creates cash drag that erodes returns vs the benchmark.
**Implementation**: Add `DRIPPolicy` model: `(id, tenant_id, portfolio_id, instrument_id, reinvestment_pct: float, fractional_allowed: bool)`. Add `register_drip_policy(portfolio_id, instrument_id, reinvestment_pct, fractional_allowed)`. When `record_corporate_action` is called with `action_type = "dividend"`, check for active DRIP policies for the affected instrument across all portfolios. For each match, compute `units_to_reinvest = floor(dividend_amount / market_price)`, call `add_holding` for the reinvested units, record a `cash_movement` for the residual cash, and emit a `drip_executed` audit event.
**Competitor**: Computershare DRIP, State Street Global Advisors DRIP Engine, FNZ Platform DRIP

---

### I15. NATS-Backed Persistent Audit Log with Event Sourcing

**Category**: Infrastructure / Compliance
**Justification**: The current audit log is an in-memory list that is lost on service restart. Regulators (CMA Kenya, CBK) and institutional auditors require immutable, queryable, off-system audit trails that can reconstruct portfolio state at any point in time (event sourcing). A NATS JetStream subject with `Limits`-based retention provides exactly this: durable, replayable, tamper-evident event storage with sub-millisecond write latency and no dependency on a separate database for the audit path.
**Implementation**: Add an optional `nats_url` parameter to `PortfolioManagementService.__init__`. When set, initialise a NATS async client with a JetStream context on startup (`await nats.connect(nats_url)`). Modify `_audit` to publish each audit record to `apg.fintech.portfolio.audit.{tenant_id}` subject with `ack_wait=5s`. Add `replay_audit_events(portfolio_id, from_sequence: int = 0)` that subscribes to the JetStream consumer and reconstructs portfolio state by replaying audit events in order — enabling point-in-time portfolio reconstruction for regulatory enquiries and dispute resolution.
**Competitor**: Kafka Event Sourcing (competitor), AWS EventBridge Archive, Axon Framework + Event Store
