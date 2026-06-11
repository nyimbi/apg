# Treasury Management — World Class Improvements

© 2025 Datacraft | Author: Nyimbi Odero

---

### I1. Real-Time Intraday Cash Position via NATS Event Sourcing

**Category**: Architecture / Cash Management
**Justification**: Current `cash_position()` does a full table scan of `treasury_postings` on every call (up to 500K rows). Event-sourced intraday positions maintained in-memory via NATS JetStream subjects reduce query latency from O(N) to O(1) and enable sub-second position updates as postings arrive.
**Implementation**: Publish a `treasury.postings.{entity_id}.{currency}` NATS subject on each posting. A bytewax dataflow subscribes, maintains a running balance per (entity, currency) in a state store, and snapshots to PostgreSQL every 60 seconds. `cash_position()` reads the snapshot + delta from NATS rather than scanning the full table.
**Competitor**: Finastra Fusion Treasury uses in-memory position engines with event-driven updates from payment gateways (sub-100ms refresh).

---

### I2. Monte Carlo VaR Engine for FX Hedge Portfolio

**Category**: Risk Analytics
**Justification**: The current `hedge_effectiveness_test()` uses a trivial dollar-offset ratio with a placeholder MTM delta. A proper Value-at-Risk engine with Monte Carlo simulation across 10,000 paths gives regulators and ALCO a statistically defensible risk number with confidence intervals, tail risk, and conditional VaR (CVaR).
**Implementation**: Add `fx_portfolio_var()` method. Use numpy to simulate FX rate paths from historical volatility and correlation matrices stored in PostgreSQL. Compute 1-day and 10-day VaR at 95% and 99% confidence. Persist simulation seeds for auditability. Stream results to dashboard via NATS `treasury.risk.var.{entity_id}`.
**Competitor**: Murex MX.3 includes a full Monte Carlo VaR engine for FX, rates, and credit with FRTB compliance.

---

### I3. ALCO Committee Decision Workflow with Four-Eyes Approval Chain

**Category**: Governance
**Justification**: Asset-Liability Committee (ALCO) decisions currently have no workflow engine — there is no concept of a motion, quorum, vote, or binding resolution. World-class treasury systems enforce maker-checker-approver chains with time-stamped digital signatures, quorum tracking, and immutable audit records for regulatory scrutiny.
**Implementation**: Add `alco_motion_create()`, `alco_motion_vote()`, `alco_resolution_finalize()` methods. Store motions in PostgreSQL with participant set, quorum threshold, vote records, and outcome. Publish `treasury.alco.motion.{id}` NATS events to notify participants. Block limit changes and policy updates until an ALCO resolution is attached.
**Competitor**: FIS Integrity Treasury requires formal ALCO approval records linked to dealing mandate changes before they take effect.

---

### I4. Dynamic Liquidity Coverage Ratio (LCR) Calculator with HQLA Buffer Tracking

**Category**: Regulatory Compliance
**Justification**: The current `regulatory_capital_report()` uses a rough proxy (`total_assets * 0.65`) for LCR. CBK requires daily LCR reporting with actual High Quality Liquid Asset (HQLA) classification (Level 1, 2A, 2B), net cash outflow calculation with Basel III stress haircuts, and a 30-day survival horizon report.
**Implementation**: Add `lcr_daily_calculation()` that queries the securities ledger for HQLA inventory, applies regulatory haircuts by asset class, computes net stressed outflows from deposit run-off rates and committed facilities, and returns LCR as HQLA / net_outflows × 100. Alerts if LCR < 100% (regulatory minimum) or < 120% (internal buffer).
**Competitor**: Oracle OFSA and Moody's RiskCalc both offer granular LCR engines mapping balance sheet line items to Basel III buckets.

---

### I5. Yield Curve Construction and Interest Rate Sensitivity (DV01/BPV) Engine

**Category**: Market Risk
**Justification**: The existing `interest_rate_risk_report()` computes a rough BPV from average rate with no curve construction. A proper yield curve (bootstrapped from KIBOR fixings, T-bill auctions, and bond market data) enables accurate DV01 per bucket, key rate duration, and parallel/twist/butterfly scenario analysis across the full term structure.
**Implementation**: Add `yield_curve_construct()` and `dv01_ladder_report()` methods. Bootstrap spot rates from money market rates stored via `benchmark_rate_submission()`. Compute DV01 for each instrument bucket (ON, 1W, 1M, 3M, 6M, 1Y, 2Y). Publish curve snapshots to NATS `treasury.curves.{entity_id}.{curve_type}` for downstream consumers.
**Competitor**: Bloomberg TOMS and Openlink Endur maintain multi-curve frameworks (OIS discounting + forward curves) for precise rate risk attribution.

---

### I6. Automated Nostro Reconciliation with Matching Engine

**Category**: Operations / Settlement
**Justification**: Unreconciled nostro balances are a primary source of operational risk and liquidity leakage. A matching engine that auto-reconciles SWIFT MT940/MT950 statements against internal ledger postings — flagging breaks, timing differences, and genuine mismatches — eliminates overnight manual reconciliation that currently takes hours.
**Implementation**: Add `nostro_statement_import()` and `nostro_reconciliation_run()` methods. Parse MT940 statements (structured as dicts), match against `treasury_postings` by amount, value date, and reference. Classify breaks as: matched, timing_difference, unmatched_bank, unmatched_book. Publish unmatched items to NATS `treasury.reconciliation.breaks.{account}` for investigation workflow.
**Competitor**: SmartStream TLM Reconciliations and SWIFT Accord process millions of nostro matches daily with <1% break rate via multi-pass fuzzy matching.

---

### I7. FX Options Pricing (Black-Scholes / Garman-Kohlhagen) with Greeks

**Category**: Derivatives Pricing
**Justification**: `hedge_instrument_create()` accepts FX options but has no pricing model — `fair_value` defaults to 0.0. World-class treasury systems compute option fair value, delta, gamma, vega, theta, and rho at booking and on each MTM cycle, enabling dynamic delta hedging and P&L attribution.
**Implementation**: Add `fx_option_price()` method using the Garman-Kohlhagen closed-form formula (generalisation of Black-Scholes for currency options). Inputs: spot, strike, domestic/foreign rate, implied vol, tenor. Returns fair value + full Greeks dict. Integrate with `hedge_effectiveness_test()` so options use theoretical delta for effectiveness ratio rather than realized MTM.
**Competitor**: Refinitiv (LSEG) Eikon and Bloomberg OVML compute full option chains with vol surface interpolation and smile calibration.

---

### I8. Cash Flow at Risk (CFaR) with AR/AP Schedule Integration

**Category**: Liquidity Risk
**Justification**: The `liquidity_forecast()` method currently produces zero inflows and outflows — all projections are empty placeholders. CFaR quantifies the range of cash flow outcomes under uncertainty by combining AR/AP payment schedules with probabilistic payment timing distributions, giving a P5–P95 confidence band rather than a single point estimate.
**Implementation**: Add `cashflow_at_risk()` method. Pull AR/AP schedules (simulated via ERP adapter pattern). Apply log-normal payment timing distributions parameterised from historical payment behaviour. Run 1,000 simulations. Return P5, P25, P50, P75, P95 percentile cash flows per day. Store distribution parameters for backtesting.
**Competitor**: GTreasury and Kyriba both offer statistical cash flow forecasting with variance-based confidence intervals linked to ERP AR/AP data.

---

### I9. Multi-Entity Cash Pooling with Overlay Structure and In-Pool Interest Allocation

**Category**: Cash Management
**Justification**: The current `cash_pooling()` supports only flat physical/notional pooling with no interest allocation logic. Enterprise treasury structures use tiered overlay pools (header → sub-header → participant) with inter-pool credit limits, in-pool interest rates differentiated by tier, and automatic sweep order (sweep deficits before surpluses to minimise borrowing cost).
**Implementation**: Add `cash_pool_configure()` and `in_pool_interest_allocate()` methods. Store pool hierarchy in PostgreSQL adjacency list. On each sweep, traverse the tree bottom-up, compute net position at each node, apply tiered rates, and generate interest allocation entries. Publish pool state to NATS `treasury.pools.{pool_id}.sweeps`.
**Competitor**: Citi Treasury and Trade Solutions (TTS) and Deutsche Bank Autobahn support multi-tiered notional pooling with real-time interest optimisation for multinationals with 100+ participating accounts.

---

### I10. Transfer Pricing Arm's-Length Rate Engine with Comparable Uncontrolled Price (CUP) Method

**Category**: Tax / Compliance
**Justification**: `transfer_pricing_report()` uses a hardcoded 7.5% arm's-length rate. OECD BEPS Action 4 and KRA requirements demand that intercompany loan rates be benchmarked against actual market data using the Comparable Uncontrolled Price method — matching currency, tenor, credit rating, and seniority. A static rate creates tax exposure.
**Implementation**: Add `transfer_pricing_benchmark_rate()` method. Query the `benchmark_rate_submissions` store for market rates matching the instrument's currency and tenor. Apply a credit spread based on entity internal credit rating (stored in entity master data). Return a defensible arm's-length range (low, midpoint, high) and flag any existing loans outside the range.
**Competitor**: BearingPoint TP Manager and Deloitte TP Analytics both automate CUP benchmarking for intercompany transactions with direct OECD database connectivity.

---

### I11. Real-Time FX Position Limit Breach Detection via NATS Streaming

**Category**: Risk Controls
**Justification**: Dealer limit monitoring (`dealer_limit_monitoring()`) is a point-in-time query — it does not fire in real time as deals are booked. A streaming breach detector that evaluates every deal booking event against pre-loaded limit matrices (per currency pair, per dealer, per entity, per counterparty) and blocks or alerts within milliseconds is required for a proper dealing room control framework.
**Implementation**: Add `fx_limit_matrix_configure()` method to store multi-dimensional limit grids in PostgreSQL. Start a bytewax dataflow subscribed to `treasury.deals.booked` NATS subject. On each event, look up the applicable limit from an in-memory limit cache (refreshed from DB on change events). Publish `treasury.limits.breach.{dealer_id}` if any dimension is breached. Emit `treasury.limits.warning.{dealer_id}` at 80%.
**Competitor**: Finastra Fusion Risk and Misys Summit both offer millisecond pre-deal limit checking integrated directly into the dealing blotter workflow.

---

### I12. Net Stable Funding Ratio (NSFR) Calculator with Asset/Liability Maturity Ladder

**Category**: Regulatory Compliance
**Justification**: Alongside LCR, CBK and Basel III require NSFR reporting (available stable funding / required stable funding ≥ 100%). No NSFR calculation exists. A maturity ladder (inflows vs outflows by bucket: O/N, 1W, 1M, 3M, 6M, 1Y, >1Y) is foundational for both NSFR and overall structural liquidity management.
**Implementation**: Add `nsfr_calculation()` and `maturity_ladder_report()` methods. Classify liabilities by Available Stable Funding (ASF) factor and assets by Required Stable Funding (RSF) factor per Basel III Annex 1 tables. Pull from `mm_placements`, `intercompany_loans`, and `bank_facilities`. Return NSFR ratio, ASF total, RSF total, and the full maturity ladder with net position by bucket.
**Competitor**: Moody's Analytics BancWare and IBM OpenPages both provide regulatory NSFR engines with built-in Basel III factor tables.

---

### I13. Cross-Currency Basis Swap Pricing and Hedge Accounting Documentation Generator

**Category**: Derivatives / Accounting
**Justification**: Cross-currency basis swaps (CCBS) are the primary instrument used by East African corporates to hedge USD/KES long-term funding mismatches. No CCBS pricing exists. Additionally, IFRS 9 hedge accounting requires formal written designation documentation at inception — currently manual and inconsistently applied.
**Implementation**: Add `ccbs_price()` method computing fair value from bootstrapped USD and KES yield curves plus basis spread. Add `hedge_accounting_designation_doc()` that auto-generates an IFRS 9-compliant designation document (hedging relationship description, risk being hedged, hedging instrument, hedged item, effectiveness testing methodology, rebalancing policy) pre-populated from the hedge record.
**Competitor**: Numerix CrossAsset and FinCAD both price CCBS and generate IFRS 9 documentation packets integrated with their TMS modules.

---

### I14. Treasury Workstation AI Co-Pilot (Ollama-Backed Deal Recommendation Engine)

**Category**: AI-Augmented Treasury
**Justification**: Treasurers spend hours analysing position reports, rate sheets, and covenant certificates before making placement or hedging decisions. An LLM-backed co-pilot that ingests the current position snapshot, market rates, upcoming maturities, and covenant headroom — and proposes a ranked set of actions (e.g., "place KES 500M at KIBOR+75bps maturing 2026-09-15 to optimise WACOF") — delivers compounding value as it learns from accepted/rejected recommendations.
**Implementation**: Add `treasury_copilot_recommend()` method. Build a context blob from `treasury_kpi_dashboard()`, `interest_rate_risk_report()`, and `liquidity_forecast()`. Send to locally-hosted Ollama (`OLLAMA_BASE_URL`) with a structured prompt requesting JSON-format action recommendations ranked by expected NII improvement. Parse and persist recommendations in `treasury_copilot_recommendations` store. Emit via NATS `treasury.copilot.recommendations.{entity_id}`.
**Competitor**: ION Treasury (Openlink) and Salmon Software both ship AI assistants that suggest deal structures; they use cloud LLMs with data-residency concerns absent from a local Ollama deployment.

---

### I15. Automated SWIFT gpi Tracker Integration with Payment Certainty Dashboard

**Category**: Payments / Operations
**Justification**: `swift_message_send()` fires and forgets — there is no feedback loop on payment status. SWIFT gpi (global payments innovation) provides end-to-end payment tracking with confirmed credit timestamps, correspondent bank fee deductions, and a "stop and recall" capability. Integrating gpi tracking eliminates manual payment tracing calls and provides real-time certainty of funds receipt.
**Implementation**: Add `swift_gpi_status_check()` and `swift_gpi_recall()` methods. Store gpi UETR (Unique End-to-end Transaction Reference) on each outbound MT103. Poll (or receive webhooks from) the SWIFT gpi Connector for status updates: initiated → in_progress → credited → completed. Persist status history in `swift_gpi_tracking` store. Publish status transitions to NATS `treasury.swift.gpi.{uetr}`. Aggregate into a payment certainty dashboard via `treasury_kpi_dashboard()` extension.
**Competitor**: J.P. Morgan ACCESS and Citi Treasury provide embedded SWIFT gpi tracking dashboards with real-time credited confirmation, eliminating the 1-2 day reconciliation lag in traditional correspondent banking.
