# Robo Advisory — World-Class Improvement Roadmap

**Capability**: `fintech_robo` | **Domain**: `fintech` | **Version target**: `2.0.0`
**Author**: Nyimbi Odero | **© 2025 Datacraft** | www.datacraft.co.ke

---

## Improvement 1: Monte Carlo Retirement Simulation

**Current state**: `projected_retirement_income` uses a single deterministic compound-growth formula — no uncertainty quantification.

**Improvement**: Run N=10,000 Monte Carlo paths per asset class using historical return distributions (mean + std-dev). Return a percentile fan (P10, P25, P50, P75, P90) of projected portfolio values. This is the industry standard for retirement planning (Vanguard, Fidelity, Betterment all do this). Enables probability-of-success framing ("78% chance of reaching goal").

**Implementation**: `async def monte_carlo_retirement_simulation(profile_id, n_paths=10000)` — draw returns from `random.gauss` per asset class, sum weighted, compound. Store full distribution in-memory; return percentile summary.

---

## Improvement 2: Dynamic Asset Allocation with Lifecycle Glide Path

**Current state**: `_MODEL_ALLOCATIONS` is a static lookup table per risk profile. Allocation never evolves as the investor ages.

**Improvement**: Implement a target-date glide path engine that shifts allocation from equity-heavy to bond-heavy as the investor approaches the goal horizon. Standard lifecycle finance formula: `equity_pct = max(0, 110 - current_age)`. Allow configurable glide paths per goal type (retirement vs education vs emergency).

**Implementation**: `async def compute_glide_path_allocation(profile_id, goal_id)` — reads current age from profile metadata, computes the optimal allocation for the time remaining, returns the target allocation and the delta vs current holdings.

---

## Improvement 3: Factor-Based Portfolio Construction

**Current state**: Portfolios are allocated by broad asset class only (equities, bonds, cash). No factor exposure.

**Improvement**: Extend allocation model to track factor exposures: Value, Momentum, Quality, Low-Vol, Size (Fama-French 5-factor). Each model portfolio gets a factor loading vector. This enables factor-tilted portfolios for sophisticated investors and factor drift detection alongside allocation drift.

**Implementation**: `async def factor_exposure_analysis(profile_id)` — computes and returns Fama-French factor loadings for the current holdings mix, compares to target factor loadings, flags factor drift.

---

## Improvement 4: Real-Time Market Regime Detection

**Current state**: Expected returns (`_ASSET_RETURNS`) are hardcoded constants — never updated based on market conditions.

**Improvement**: Integrate a Hidden Markov Model (2-state: bull/bear) trained on rolling 12-month return windows. Use regime state to dynamically adjust expected return estimates: in bear regime, reduce equity return assumptions by 40%, increase bond and cash weights. This is standard in quantitative asset management.

**Implementation**: `async def market_regime_detection(lookback_days=252)` — returns current regime (bull/bear/uncertain), confidence score, and regime-adjusted return forecasts per asset class.

---

## Improvement 5: Tax-Loss Harvesting with Wash-Sale Compliance Engine

**Current state**: `tax_optimisation` uses a flat 12% stub loss assumption on all volatile assets. No wash-sale rule compliance.

**Improvement**: Track each tax lot (purchase date, cost basis, current value). Identify lots with losses >= threshold. Enforce the 30-day wash-sale window by tracking recent purchase history and automatically selecting a sufficiently different replacement instrument. Support both Kenya CGT (15%) and international jurisdictions (US, UK, DE).

**Implementation**: `async def tax_lot_harvesting_engine(profile_id, jurisdiction, min_lot_age_days=31)` — scans all lots, returns harvest candidates with wash-sale-safe replacements, estimated tax alpha, and filing reference data.

---

## Improvement 6: Behavioural Finance Nudge Engine

**Current state**: No client-facing behavioural coaching. Questionnaire responses are scored mechanically with no behavioural insight.

**Improvement**: Analyse questionnaire response patterns to detect behavioural biases: loss aversion (high loss_reaction score + low horizon), recency bias (over-weighting recent market events), overconfidence (high experience score + aggressive profile). Surface nudges via notification. Include a bias-adjusted risk profile alongside the raw score.

**Implementation**: `async def behavioural_bias_analysis(questionnaire_id)` — returns detected biases, severity scores, and evidence-based nudge messages aligned with client communication standards.

---

## Improvement 7: Multi-Currency Portfolio Valuation

**Current state**: All valuations are in USD. `currency` parameter exists but is not used in calculations.

**Improvement**: Maintain FX rate cache (fetched from Ollama-served rate model or stored ECB/CBK reference rates). Convert all asset values to the investor's base currency for reporting. Track currency exposure separately as a source of portfolio risk. Support KES, USD, EUR, GBP, NGN, GHS, ZAR natively (the already-declared `SUPPORTED_CURRENCIES`).

**Implementation**: `async def fx_adjusted_portfolio_valuation(profile_id, target_currency)` — returns holdings restated in target currency, FX PnL attribution, and currency hedge recommendations.

---

## Improvement 8: Robo-to-Human Escalation Workflow

**Current state**: No mechanism to escalate from robo to human adviser when complexity exceeds robo scope.

**Improvement**: Define escalation triggers: portfolio value > $500k, distressed goal (< 30% funded with < 2 years remaining), tax complexity (multiple jurisdictions), client behavioural flag (panic selling pattern). On trigger, generate an escalation ticket, assign a human adviser from the `fintech_wealth` pool, and freeze automated rebalancing pending human review.

**Implementation**: `async def evaluate_escalation_triggers(profile_id)` — returns escalation decision, triggering conditions, assigned adviser reference, and recommended human intervention actions.

---

## Improvement 9: Drawdown Circuit Breaker

**Current state**: Auto-invest and auto-rebalance execute on schedule regardless of market conditions.

**Improvement**: Implement a drawdown circuit breaker: if portfolio value drops > 15% from peak in 30 days, suspend auto-invest and auto-rebalance, notify client, and wait for human or client confirmation before resuming. Track portfolio high-water mark per portfolio. This prevents buying into a crash without client awareness — a regulatory requirement in several jurisdictions.

**Implementation**: `async def drawdown_circuit_breaker_check(portfolio_id, peak_value_usd)` — computes current drawdown from peak, evaluates circuit breaker condition, returns suspend/allow decision with rationale.

---

## Improvement 10: Personalised Benchmark Construction

**Current state**: Benchmarks are hardcoded to a risk profile (e.g. "balanced"). Alpha is computed vs a generic benchmark.

**Improvement**: Construct a personalised benchmark (also called a "policy portfolio") that exactly matches the investor's target allocation at the start of the measurement period. Attribution decomposes return into: allocation effect, selection effect, and interaction effect (Brinson-Hood-Beebower). This is the correct way to measure robo performance — the generic benchmark comparison is misleading for non-standard allocations.

**Implementation**: `async def brinson_attribution_report(profile_id, period_start, period_end)` — computes BHB attribution decomposition, returns allocation, selection, and interaction effects per asset class.

---

## Improvement 11: Regulatory Reporting Automation (CMA / FCA / SEC)

**Current state**: `cma_robo_return` is a stub that returns a draft report with counts only.

**Improvement**: Generate fully structured regulatory reports per jurisdiction: CMA (Kenya) Form RA-01, FCA (UK) RMAR Section J, SEC (US) Form ADV Part 2A. Each report includes: AUM by risk tier, fee disclosure breakdown, suitability assessment summary, complaint log, and system outage log. Auto-file to regulator API endpoint when available; otherwise generate signed PDF via Ollama document generation.

**Implementation**: `async def regulatory_report(jurisdiction, report_type, period)` — dispatches to jurisdiction-specific report builder, returns structured report payload and filing status.

---

## Improvement 12: Goal Sensitivity Analysis

**Current state**: `goal_tracking` computes a single point estimate for required monthly contributions.

**Improvement**: Run sensitivity analysis across three dimensions: (1) expected return ± 2%, (2) monthly contribution ± 25%, (3) time horizon ± 2 years. Return a 3×3 heatmap of probability-of-goal-achievement. This gives clients an intuitive feel for which lever matters most. Standard in financial planning software (eMoney, MoneyGuidePro).

**Implementation**: `async def goal_sensitivity_analysis(goal_id, n_scenarios=27)` — iterates parameter grid, calls goal projection for each, returns sensitivity grid and the single highest-leverage recommendation.

---

## Improvement 13: Portfolio Stress Testing

**Current state**: No stress testing. Risk is proxied by a single volatility estimate (`portfolio_return * 0.15`).

**Improvement**: Apply historical stress scenarios: 2008 Global Financial Crisis (equities -50%, bonds +10%), 2020 COVID crash (equities -35%, 3-week horizon), 2022 rate-shock (bonds -20%, equities -25%). For each scenario, compute portfolio drawdown and recovery time estimate. Compare stress loss against client's loss tolerance from questionnaire responses. Flag if stress loss > stated loss tolerance.

**Implementation**: `async def portfolio_stress_test(profile_id, scenarios=None)` — runs all named scenarios, returns per-scenario portfolio loss, max drawdown, estimated recovery months, and suitability flag.

---

## Improvement 14: Automated Dividend and Coupon Reinvestment

**Current state**: Auto-invest only handles new cash contributions. Income (dividends, coupon payments) is not tracked or reinvested.

**Improvement**: Track income events per asset class (dividend yield for equities, coupon rate for bonds). On each scheduled auto-invest run, credit income proceeds to the portfolio, then reinvest according to current target allocation. Track total income vs capital gain separately for tax reporting. This is essential for total-return accounting.

**Implementation**: `async def reinvest_income(portfolio_id, income_events=None)` — credits income from asset class yield assumptions (or explicit `income_events` list), reinvests per target allocation, returns reinvestment log entry and updated portfolio valuation.

---

## Improvement 15: Client Lifetime Value (CLV) and Churn Prediction

**Current state**: No client analytics beyond raw portfolio metrics.

**Improvement**: Compute client lifetime value: projected AUM growth × fee rate × expected tenure. Use a simple logistic regression (trained on behavioural signals in-process using scikit-learn or pure-Python) to predict churn probability: inputs are goal progress, days since last login, questionnaire completion rate, behavioural bias score. High-churn-risk clients receive an automated retention intervention (personalised goal progress email + discounted fee offer).

**Implementation**: `async def client_lifetime_value(customer_id)` and `async def churn_probability(customer_id)` — compute CLV using Gordon Growth Model variant, compute churn score from feature vector, return actionable retention recommendation.
