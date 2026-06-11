# FinTech Risk Management — World-Class Improvement Roadmap

**Capability**: `fintech_risk` | **Version target**: 2.0.0 | **Date**: 2026-06-11

---

## 1. Historical VaR Backtesting (Kupiec POF Test)

**Current gap**: `market_risk_var` uses parametric VaR with no backtesting validation.

**Improvement**: Implement the Kupiec Proportion of Failures (POF) test and Christoffersen interval-forecast test over the rolling return window. Track VaR exceedances and emit a `var_backtest_exception` risk event when the POF p-value falls below 0.05. This transforms VaR from a point estimate into a statistically validated forecast with model-risk controls that satisfy BCBS 239 model risk governance requirements.

---

## 2. Monte Carlo CVaR (Conditional Value-at-Risk / Expected Shortfall)

**Current gap**: Only parametric VaR is computed; Basel IV / FRTB replaces VaR with Expected Shortfall (ES) at 97.5%.

**Improvement**: Add `async market_risk_cvar()` using Monte Carlo simulation (N=10,000 paths) with Cholesky-decomposed correlated asset returns. Return ES at 97.5% alongside VaR, plus the full loss distribution tail. The simulation should be offloaded via `asyncio.to_thread` to avoid blocking the event loop.

---

## 3. Dynamic Probability of Default Calibration (Merton Structural Model)

**Current gap**: PD estimates are static proxies (`risk_score / 100 * 0.15`) with no market-implied calibration.

**Improvement**: Implement the Merton (1974) structural model: derive PD from asset value (V_A), asset volatility (σ_A), and debt face value (D) using Black-Scholes equations. For listed counterparties, ingest equity price and volatility; for unlisted, fall back to Altman Z-score derived from balance sheet ratios. Output point-in-time PD alongside through-the-cycle PD for IFRS 9 staging accuracy.

---

## 4. IFRS 9 Three-Stage Bucket Migration Engine

**Current gap**: `ecl_computation` assigns a static IFRS 9 stage but does not track stage transitions over time or apply forward-looking macro overlays.

**Improvement**: Build a stage migration engine that snapshots each profile's stage at every reporting date, detects Significant Increase in Credit Risk (SICR) triggers (30-day past due, credit watch, macro threshold breaches), and applies probability-weighted macro scenarios (base / adverse / optimistic) using an overlay multiplier per CBK Prudential Guideline CBK/PG/01. Persist the migration history for auditor traceability.

---

## 5. Intraday Liquidity Monitoring (BCBS 248)

**Current gap**: `liquidity_risk_report` produces a static LCR/NSFR snapshot; no intraday settlement position tracking.

**Improvement**: Add `async intraday_liquidity_monitor()` that maintains a real-time settlement position ledger per correspondent bank, tracks peak intraday liquidity usage, and triggers early-warning alerts when intraday usage exceeds 80% of the available intraday limit. This satisfies BCBS 248 intraday liquidity monitoring requirements and CBK supervision expectations.

---

## 6. Regulatory Capital Optimizer (Basel IV SA-CR)

**Current gap**: RWA computation uses flat weights; does not reflect Basel IV Standardised Approach credit risk (SA-CR) risk weights or FRTB Standardised Approach (SA) market risk capital.

**Improvement**: Implement the full Basel IV SA-CR risk-weight lookup table (sovereigns, banks, corporates, retail, SME, real estate by LTV band, defaulted exposures) and the FRTB Sensitivity-Based Approach (SBA) for market risk capital. Output a capital breakdown: credit RWA, market RWA, operational RWA (BIA/SMA), plus the aggregate CET1, AT1, and T2 capital stack required.

---

## 7. Concentration Risk via DRC Granularity Adjustment (FRTB)

**Current gap**: HHI is the only concentration measure; no issuer/sector/country concentration charges.

**Improvement**: Add Default Risk Charge (DRC) granularity adjustment per FRTB rules: compute JTD (Jump-to-Default) per issuer, apply net-long/net-short netting within buckets, and calculate the DRC add-on. Supplement with sector HHI (using GICS Level 2 sectors) and country concentration ratio (CR3, CR5). This enables FRTB-compliant capital allocation for trading book portfolios.

---

## 8. Real-Time AML Graph Analytics (Entity Resolution)

**Current gap**: `aml_transaction_monitoring` operates on individual transactions with no cross-transaction entity linkage.

**Improvement**: Maintain an in-memory transaction graph (networkx DiGraph or equivalent) where nodes are entities (accounts, counterparties, beneficial owners) and edges are transactions. Run graph centrality metrics (PageRank, betweenness) on the rolling 90-day window to surface high-centrality nodes indicative of layering networks. Flag structuring rings using connected-component analysis with a configurable minimum ring size. Emit FATF R.16 wire transfer alerts for missing originator information.

---

## 9. Behavioral Scoring with LSTM Anomaly Detection

**Current gap**: `risk_scoring_model_run` uses a deterministic rule-based score with no temporal pattern learning.

**Improvement**: Implement an LSTM autoencoder (via PyTorch, served via local Ollama or ONNX Runtime for CPU inference) trained on normal transaction sequences per customer cohort. Reconstruction error above a threshold triggers a behavioral anomaly alert. Inputs: normalized transaction amounts, merchant category codes, time-of-day, device fingerprint hash. Output: anomaly score 0–100 alongside contributing feature SHAP values for explainability under CBK Consumer Protection guidelines.

---

## 10. Stress Testing via Reverse Stress Test Engine

**Current gap**: `stress_test_portfolio` applies a single forward shock; no reverse stress testing.

**Improvement**: Add `async reverse_stress_test()` that finds the minimum shock magnitude sufficient to breach a capital or liquidity threshold. Use bisection search over shock_bps space [0, 10000] to locate the tipping point within 20 iterations. Output the critical shock, the binding constraint (CAR, LCR, NSFR, or VaR), and the implied scenario narrative. This is an ICAAP / ILAAP Board-level stress test tool.

---

## 11. Watchlist & Sanctions Screening Engine

**Current gap**: No integrated sanctions screening; AML signals depend on transaction flags only.

**Improvement**: Add `async sanctions_screening()` that checks subjects against OFAC SDN, EU Consolidated List, UN Consolidated List, and CBK Designated Entities using fuzzy name matching (Jaro-Winkler distance ≥ 0.92) with alias expansion. Cache list snapshots with a 24-hour TTL. Return match confidence score, matched list, matched entry, and recommended action (block / hold / EDD). Integrate into `create_profile` as a pre-condition check.

---

## 12. Risk-Adjusted Return on Capital (RAROC) Calculator

**Current gap**: No profitability-adjusted risk metrics; decisions are purely loss-focused.

**Improvement**: Add `async raroc_calculation()` that computes RAROC = (Net Revenue - Expected Loss - Allocated OpEx) / Economic Capital, where Economic Capital = UL × confidence multiplier (2.33 for 99%). Output RAROC per product, portfolio, and customer segment with a hurdle rate comparison (configurable, default 15% for Kenya market). Enable product managers to identify risk-adjusted profitable segments and price new products correctly.

---

## 13. Automated Regulatory Report Generation (CBK/CMA Schedules)

**Current gap**: `export_risk_data` produces a raw data dump with no regulatory formatting.

**Improvement**: Add `async generate_regulatory_report()` supporting CBK Prudential Return formats: PR1 (capital adequacy), PR2 (large exposures), PR3 (liquidity), PR4 (asset quality), and CMA periodic risk disclosure. Generate structured JSON/CSV/Excel outputs per the published CBK template specifications with auto-validation of mandatory fields and cross-schedule consistency checks. Include a digital signature hash for submission integrity.

---

## 14. Model Risk Management Framework (SR 11-7 / SS1/23 Compliance)

**Current gap**: `model_validation_report` produces a synthetic accuracy metric with no lifecycle governance.

**Improvement**: Implement full model risk management lifecycle: model inventory (registration, version, owner, purpose, materiality tier), pre-deployment validation gate (discriminatory power: Gini ≥ 0.35, KS ≥ 0.25; calibration: Hosmer-Lemeshow p > 0.05; stability: PSI < 0.10), ongoing monitoring with monthly PSI/CSI computation, and annual revalidation triggers. Emit `model_drift` risk events automatically when PSI > 0.10. Satisfies PRA SS1/23 and Fed SR 11-7 guidance.

---

## 15. Integrated Risk Appetite Dashboard with RAG Status

**Current gap**: `risk_appetite_monitoring` returns raw utilisation data with no aggregated organisational RAG (Red/Amber/Green) status.

**Improvement**: Build a hierarchical RAG status engine that aggregates appetite utilisation from transaction-level to desk, business unit, and board level using a configurable tolerance band (Green < 70%, Amber 70–90%, Red > 90%). Produce a board-ready risk appetite statement PDF with trend sparklines per domain, breach history, forward-looking projections using ARIMA extrapolation on 90-day utilisation series, and automated narrative generation via local Ollama LLM (mistral or llama3). Persist RAG snapshots for longitudinal trend analysis.
