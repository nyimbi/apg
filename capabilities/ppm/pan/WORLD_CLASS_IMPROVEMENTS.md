# Portfolio Analytics (ppm_pan) — World-Class Improvements

**Capability**: `ppm_pan` | **Domain**: `ppm` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Earned Value Management (EVM) Integration

**Current gap**: Performance snapshots capture a single actual vs. benchmark value; no SPI/CPI/EAC/TCPI are computed.

**Improvement**: Add `async earned_value_metrics(portfolio_id, as_of_date)` that computes Schedule Performance Index (SPI), Cost Performance Index (CPI), Estimate at Completion (EAC), To-Complete Performance Index (TCPI), and Schedule Variance (SV) / Cost Variance (CV) for every project in the portfolio, then aggregates to portfolio level. These are the gold-standard delivery health indicators used by PMI and NATO defence programmes.

---

## 2. Monte Carlo Schedule & Cost Risk Simulation

**Current gap**: `portfolio_optimisation` uses deterministic greedy knapsack; no uncertainty modelling.

**Improvement**: Add `async monte_carlo_risk_simulation(portfolio_id, iterations, confidence_levels)` that samples each project's cost and duration from triangular distributions (optimistic / most-likely / pessimistic inputs), runs N iterations, and returns percentile outcomes (P50, P80, P90) for portfolio cost-at-completion and schedule-at-completion. Surfaces the "cone of uncertainty" that executives need for contingency budgeting.

---

## 3. Rolling Wave Benefits Realisation Forecast

**Current gap**: `benefits_realisation_tracking` records one actuals-vs-plan entry; it does not project future benefit curves.

**Improvement**: Add `async benefits_realisation_forecast(project_id, forecast_periods)` that fits a logistic S-curve to the observed actuals history and extrapolates the remaining benefit realisation trajectory. Outputs period-by-period forecast with confidence bands and a "benefits at risk" flag when cumulative realisation lags the planned profile by > 10%.

---

## 4. Strategic Portfolio Bubble Chart Data

**Current gap**: The risk-return matrix returns raw data points; no quadrant interpretation or bubble sizing.

**Improvement**: Add `async portfolio_bubble_chart(portfolio_id, x_metric, y_metric, size_metric)` that returns fully normalised bubble chart data where x, y, and bubble size can be any combination of risk_score, return_value, alignment_score, budget, progress_pct, or demand_fte. Enables executives to swap axes on the fly without re-querying the server.

---

## 5. Delivery Velocity Trending

**Current gap**: No time-series analysis of portfolio throughput.

**Improvement**: Add `async delivery_velocity_trend(portfolio_id, window_weeks)` that tracks project completions per rolling time window, computes a velocity trend line (linear regression), and flags portfolios where velocity is declining for ≥ 2 consecutive windows. Provides the "factory output" view that PMO directors need for capacity planning.

---

## 6. Cross-Portfolio Dependency Risk Map

**Current gap**: Projects within portfolios are tracked independently; inter-portfolio dependencies are invisible.

**Improvement**: Add `async cross_portfolio_dependency_map(tenant_id)` that ingests declared project-to-project dependencies, computes a directed dependency graph, identifies critical path segments that span portfolio boundaries, and scores each cross-portfolio link by combined risk exposure. Outputs an adjacency matrix suitable for D3 force-directed graph rendering.

---

## 7. Portfolio Balance Score (McKinsey Three Horizons)

**Current gap**: No classification of projects into innovation horizons.

**Improvement**: Add `async portfolio_balance_score(portfolio_id)` that classifies projects into Horizon 1 (run-the-business), Horizon 2 (growth), and Horizon 3 (transformation) based on their strategic_fit and innovation_index alignment scores. Computes the portfolio's investment split across horizons and benchmarks it against configurable targets (e.g., 70/20/10). Returns a balance deviation score and recommended rebalancing moves.

---

## 8. Resource Bottleneck Detector

**Current gap**: Capacity demand chart gives aggregate FTE gap but does not identify specific skill/role bottlenecks.

**Improvement**: Add `async resource_bottleneck_detector(portfolio_id, period)` that disaggregates FTE demand by role/skill across all projects, computes utilisation per role bucket, and surfaces the top-N over-allocated roles with a severity score = (demand / supply) * impact_weight. Feeds directly into the capacity heat map with bottleneck annotations.

---

## 9. Portfolio Value at Risk (VaR) Calculator

**Current gap**: Risk scoring is qualitative (category label + numeric score); no financial VaR metric.

**Improvement**: Add `async portfolio_value_at_risk(portfolio_id, confidence_pct)` that aggregates financial exposures across portfolio projects, models correlation between risk events using a Cholesky decomposition of a risk correlation matrix, and computes the portfolio VaR at the requested confidence level (e.g., 95%). Returns expected loss, worst-case loss, and diversification benefit from holding multiple projects.

---

## 10. Automated Red-Amber-Green (RAG) Escalation Engine

**Current gap**: RAG status is computed on-demand in `portfolio_health_dashboard`; no automated escalation or notification.

**Improvement**: Add `async rag_escalation_check(tenant_id, escalation_rules)` that evaluates configurable escalation rules (e.g., "if a portfolio stays RED for > 2 consecutive snapshots, create an escalation ticket and notify the portfolio owner"). Outputs a list of triggered escalation actions with stakeholder routing and evidence references, suitable for feeding into the `ntfy` capability.

---

## 11. Benchmark Gap Analysis

**Current gap**: `snapshot_performance` records one actual-vs-benchmark pair; no gap analysis across benchmark types.

**Improvement**: Add `async benchmark_gap_analysis(portfolio_id, benchmark_types)` that compares portfolio performance snapshots across multiple benchmark types simultaneously (industry_average, peer_group, historical, target, best_in_class), computes normalised gap scores, and ranks improvement opportunities by impact-to-effort ratio. Returns a waterfall data structure suited for executive gap review sessions.

---

## 12. Scenario Sensitivity Analysis

**Current gap**: `run_scenario` records a single scenario with fixed assumptions; no sensitivity testing.

**Improvement**: Add `async scenario_sensitivity_analysis(scenario_id, variable_ranges)` that performs one-at-a-time (OAT) sensitivity analysis by independently varying each input assumption within its specified range and recording the resulting change in projected outcome. Outputs a tornado chart data structure ranked by sensitivity magnitude, showing which assumptions most drive outcome variability.

---

## 13. Portfolio Lifecycle Stage Tracker

**Current gap**: Portfolio status is a single enum field; no lifecycle stage transition history.

**Improvement**: Add `async advance_portfolio_lifecycle(portfolio_id, target_stage, evidence_reference)` that validates legal stage transitions (proposed → approved → active → under_review → closed), records a full transition history with actor, timestamp, and evidence reference, and enforces approval gates between stages. Returns the updated lifecycle history as a timeline suitable for audit and governance reporting.

---

## 14. AI-Powered Portfolio Narrative Generator

**Current gap**: Reports output structured JSON; no prose narrative for executive consumption.

**Improvement**: Add `async generate_portfolio_narrative(portfolio_id, period, style)` that takes the structured output from `executive_portfolio_report` and sends it to a locally-hosted Ollama model (e.g., llama3.1:8b) with a narrative generation prompt. Outputs a 3-paragraph executive summary in plain English suitable for board packs, with configurable style ("formal", "concise", "risk-focused"). Falls back gracefully when OLLAMA_BASE_URL is not set.

---

## 15. Composability Bridge to Intel Domain

**Current gap**: Portfolio Analytics and Intelligence (intel) domain produce parallel dashboards with no cross-domain data flow.

**Improvement**: Add `async sync_to_intel_domain(portfolio_id, intel_service)` that pushes a structured portfolio health snapshot (RAG status, alignment score, EVM metrics, top risks) into the intel domain's alert and threat detection pipeline. This closes the loop between project portfolio health and enterprise intelligence: a portfolio that goes RED automatically surfaces as a monitored signal in the intel threat landscape. Requires `intel_service` to be a duck-typed dependency accepting `ingest_portfolio_signal(payload)`.
