# Cash Management — World-Class Improvement Catalogue

© 2025 Datacraft | Author: Nyimbi Odero

---

## 1. Probabilistic Cash Flow Forecasting with Confidence Intervals

**Category**: Forecasting / AI

**Justification**: Current `liquidity_forecast` returns a single deterministic net figure.
Treasury decisions require quantile distributions — knowing the P5/P50/P95 outcomes
prevents under-capitalisation and over-investment simultaneously.

**Implementation**: Monte Carlo simulation over historical cash-flow volatility per category.
Store `forecast_distribution` dict with percentiles alongside the point estimate.
Integrate with `create_cash_forecast` confidence_score field and expose as
`async probabilistic_forecast(days, simulations, tenant_id)`.

**Competitor Reference**: Kyriba AI Predict, HighRadius CashApplication

---

## 2. Real-Time Bank Feed via Open Banking / ISO 20022

**Category**: Bank Connectivity

**Justification**: `import_bank_statement` is batch-driven. Intraday liquidity management
requires near-real-time balance and transaction feeds. Under Basel III LCR monitoring,
intraday positions must be tracked.

**Implementation**: Async webhook receiver that accepts ISO 20022 camt.052 intraday
messages and camt.053 end-of-day. Parse `<Ntry>` elements, upsert positions, and
emit `intraday_position_updated` events on the Bytewax stream.

**Competitor Reference**: Finastra Fusion Cash Management, SAP S/4HANA Treasury

---

## 3. Multi-Currency Netting Engine

**Category**: FX / Intercompany

**Justification**: `intercompany_settlement` records bilateral settlements individually.
A netting engine calculates multilateral net positions across entities and currencies,
reducing settlement count by 60-80% and cutting FX conversion costs.

**Implementation**: Collect all intercompany payables/receivables per entity-pair-currency
triplet over a netting cycle, compute net matrix using transpose-elimination, generate
minimal settlement instructions, and record a `netting_cycle` result.

**Competitor Reference**: Kyriba Netting, SAP In-House Cash

---

## 4. Liquidity Coverage Ratio (LCR) and Net Stable Funding Ratio (NSFR)

**Category**: Regulatory / Risk

**Justification**: `regulatory_reporting_package` computes a `lcr_proxy` but does not
implement the full Basel III methodology — HQLA tiering (Level 1/2A/2B) and the 30-day
stress-outflow calculation. Regulators (CBK, Bank of Uganda) require granular LCR returns.

**Implementation**: Classify accounts/investments by HQLA tier. Apply haircuts per Basel
III Annex 1. Compute stressed outflows by run-off rate category. Return structured LCR
and NSFR ratios with HQLA decomposition.

**Competitor Reference**: Wolters Kluwer OneSumX, Moody's Analytics FERMAT

---

## 5. Automated Payment Factory with Priority Queuing

**Category**: Payments / Operations

**Justification**: `validate_payment_run` checks funding adequacy but does not sequence
or optimise payment release. A payment factory prioritises RTGS/high-value payments,
batches low-value ACH, and defers discretionary payments when intraday liquidity is tight.

**Implementation**: `async schedule_payment_run(run_id, priority, cutoff_time)` — assigns
priority score (urgent/normal/deferred) based on amount, payment type, and available
liquidity. Returns ordered payment schedule with estimated settlement times.

**Competitor Reference**: TIS (Treasury Intelligence Solutions), Bottomline Technologies

---

## 6. Cash Flow Categorisation via NLP / Pattern Matching

**Category**: AI / Data Quality

**Justification**: `record_cash_flow` requires the caller to supply `category`. In practice,
bank narrations are free-text. Automatic categorisation using regex patterns and optional
local LLM (Ollama) inference reduces manual coding effort and improves forecast accuracy.

**Implementation**: `async categorise_cash_flow(description, amount, account_id)` — applies
a priority-ordered rule set (regex → keyword → Ollama mistral inference) and returns
`suggested_category` with confidence. Fallback to `uncategorised` if below threshold.

**Competitor Reference**: Plaid Transactions, HighRadius AI Categorisation

---

## 7. Concentration Risk Monitoring

**Category**: Risk Management

**Justification**: `gaap_disclosure_note` flags concentration risk as a boolean. A proper
monitor tracks the percentage of total cash held at each bank, raises alerts above
configurable thresholds (e.g. >30% with single counterparty), and integrates with
`bank_covenant_compliance`.

**Implementation**: `async concentration_risk_report(as_of_date, threshold_pct)` —
computes per-bank and per-currency concentration ratios, compares to policy thresholds,
returns ranked risk findings and recommended redistributions.

**Competitor Reference**: Bloomberg BTCA, Reval (ION)

---

## 8. Automated Bank Reconciliation with Fuzzy Matching

**Category**: Reconciliation

**Justification**: `auto_reconcile_statement` matches strictly on amount string equality.
Real-world statements have timing differences, bank charges added to transactions, and
description-only references. Fuzzy matching on amount tolerance + date window + reference
similarity dramatically improves straight-through reconciliation rates.

**Implementation**: `async fuzzy_reconcile(statement_id, date_window_days, amount_tolerance_pct)`
— implements three-pass matching: exact amount+reference, amount-within-tolerance+date,
description-similarity above 0.85 cosine threshold. Returns confidence score per match.

**Competitor Reference**: BlackLine, Trintech Cadency

---

## 9. Treasury Investment Optimisation

**Category**: Investment / Optimisation

**Justification**: `create_treasury_investment` records investments but provides no guidance
on optimal allocation. A solver that maximises yield subject to liquidity constraints
(minimum cash buffer, maturity ladder, counterparty limits) improves returns by 20-40bps.

**Implementation**: `async optimise_investment_portfolio(available_cash, horizon_days,
constraints)` — formulates a linear programme (scipy.optimize or OR-Tools), returns
recommended allocation across money market, T-bills, and CPs with expected yield.

**Competitor Reference**: Clearwater Analytics, Charles River IMS

---

## 10. Working Capital Cycle Analytics

**Category**: Analytics / Working Capital

**Justification**: `working_capital_analysis` computes a cash ratio. CFOs need Days Sales
Outstanding (DSO), Days Payable Outstanding (DPO), and the Cash Conversion Cycle (CCC)
to identify working capital inefficiencies and target improvements.

**Implementation**: `async working_capital_cycle(period, tenant_id)` — derives DSO from AR
flows (collections category), DPO from AP flows (payments category), DIO from inventory
proxies. Returns CCC = DSO + DIO - DPO with period-over-period trending.

**Competitor Reference**: Esker, Serrala, HighRadius

---

## 11. Cash Flow Anomaly Detection

**Category**: Fraud / Risk

**Justification**: No current method detects unusual cash movements. Statistical anomaly
detection (z-score, IQR, or Isolation Forest via Ollama embeddings) flags flows that
deviate significantly from historical norms — protecting against fraud and input errors.

**Implementation**: `async detect_anomalies(period, tenant_id, sensitivity)` — computes
per-category rolling mean/std over trailing 90 days. Flags flows where z-score > sensitivity
threshold. Returns ranked anomaly list with deviation metrics and recommended actions.

**Competitor Reference**: Featurespace ARIC, DataVisor

---

## 12. Notional Cash Pooling with Interest Optimisation

**Category**: Cash Pooling / Group Treasury

**Justification**: `cash_pooling_sweep` executes zero-balance sweeps but does not calculate
notional pool interest — the key mechanism by which group treasuries offset overdraft
interest against credit balances without physical movement.

**Implementation**: `async notional_pool_interest(pool_id, value_date, debit_rate,
credit_rate)` — aggregates net pool balance, allocates notional interest credit/debit
to each participant proportional to their balance contribution, returns pool interest
statement with per-entity breakdown.

**Competitor Reference**: Citi Treasury and Trade Solutions, Deutsche Bank Autobahn

---

## 13. SWIFT gpi Payment Tracking Integration

**Category**: Payments / Transparency

**Justification**: Cross-border payment runs have no settlement visibility. SWIFT gpi (global
payments innovation) provides end-to-end transaction tracking via UETR (Unique End-to-end
Transaction Reference), enabling real-time confirmation and deduction of correspondent
bank charges.

**Implementation**: `async track_swift_payment(uetr, payment_run_id)` — calls SWIFT gpi
Tracker API (or mock), updates payment_run status through `processing → credited → settled`
lifecycle, records correspondent bank deductions as fee flows.

**Competitor Reference**: SWIFT gpi, Temenos Payments Hub

---

## 14. Multi-Entity Consolidated Treasury Dashboard

**Category**: Reporting / Group Treasury

**Justification**: `dashboard_summary` is single-tenant. Group CFOs managing 10-50 entities
need a consolidated view with drill-down, intra-group eliminations, and FX-translated totals
in a functional currency — currently not supported.

**Implementation**: `async consolidated_dashboard(entity_ids, functional_currency,
fx_rates, tenant_id)` — aggregates balances and flows across entities, translates to
functional currency using spot or provided rates, eliminates intercompany balances,
returns group-level and entity-level views.

**Competitor Reference**: Kyriba Global TMS, FIS Quantum

---

## 15. ESG Cash Management Reporting

**Category**: Sustainability / Compliance

**Justification**: Institutional investors and regulators increasingly require ESG-aligned
treasury disclosure. Tracking the proportion of cash held in ESG-rated banks and green
deposits, and computing the carbon footprint of payment processing, is a differentiator
for forward-looking CFOs.

**Implementation**: `async esg_treasury_report(period, tenant_id)` — classifies banks by
ESG tier (A/B/C/unrated), computes proportion of deposits in ESG-A banks, aggregates
green deposit exposures, estimates Scope 3 payment emissions using payment count × average
emission factor. Returns structured ESG disclosure aligned with TCFD recommendations.

**Competitor Reference**: Clarity AI, Sustainalytics TMS Integration
