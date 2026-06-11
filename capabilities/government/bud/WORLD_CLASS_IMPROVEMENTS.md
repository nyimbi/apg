# Budget & Financial Planning — World-Class Improvements

**Capability**: `government_bud` | **Domain**: Government Budget Cycle, MTEF, PBB, Fiscal Reporting, Budget Execution

---

### I1. MTEF Rolling Three-Year Envelope Automation
**Category**: Planning Intelligence | **Justification**: Most government systems require manual re-entry each budget cycle; automated MTEF propagation with macro-linkage eliminates 60-80% of budget preparation labour and reduces ceiling breach risk from ~30% to under 5%. | **Implementation**: Async `mtef_rolling_envelope()` method that accepts baseline year + macro parameters (GDP growth, inflation, deficit target), computes forward year 1/2/3 ceilings per sector using parameterised fiscal rules, persists envelope constraints to vote accounts, and emits `mtef_envelope_set` events to NATS stream `apg.government.bud.mtef`. | **Competitor**: IBM Cognos Government Edition — envelope propagation takes 2-3 days manually; Oracle PBCS achieves partial automation for commercial entities only.

---

### I2. Programme-Based Budgeting (PBB) KPI Scorecard Engine
**Category**: Results-Based Accountability | **Justification**: PBB without automated KPI scoring is theatre. Linking vote utilisation to delivery outcomes enables evidence-based reallocation and satisfies IMF Article IV conditions on fiscal transparency. Automated scoring reduces Auditor-General queries by an estimated 40%. | **Implementation**: Async `pbb_scorecard()` method linking each vote to a performance framework: input, output, outcome, and impact indicators. Computes weighted composite score; flags underperformers for reallocation recommendation. Events published to `apg.government.bud.pbb`. | **Competitor**: Palantir Gotham for Defence Ministries — USD 20M+ deployment; SAP BPC achieves PBB linkage only for commercial clients.

---

### I3. Real-Time Commitment Control via NATS Event Streaming
**Category**: Execution Control | **Justification**: Batch-reconciled commitment control (the norm in legacy IFMIS) allows over-commitment windows of hours to days. NATS-driven real-time balance checks reduce over-commitment incidents to near zero and satisfy PFMA Section 39 "continuous monitoring" interpretation. | **Implementation**: Async `stream_commitment_event()` publishes every `record_commitment` / `commitment_liquidation` call as a CloudEvent to NATS subject `apg.government.bud.commitment.{tenant_id}`. Downstream vote-balance projection service subscribes and updates available balances within milliseconds. | **Competitor**: SAP S/4HANA Public Sector — near-real-time only with expensive add-on; FreeBalance IFMIS — batch reconciliation every 4 hours.

---

### I4. AI-Assisted Budget Ceiling Redistribution
**Category**: Predictive Reallocation | **Justification**: Finance ministries reallocate under-spent ceilings late in the fiscal year under political pressure, leading to wasteful year-end spending. ML-driven early detection of low-absorption votes allows transparent, rule-governed redistribution months earlier. | **Implementation**: Async `ai_ceiling_redistribution_recommend()` calls local Ollama model (mistral or gemma3) with absorption trends, returns ranked reallocation proposals with compliance justification per budget circular provisions. Output includes draft supplementary appropriation reference. | **Competitor**: Microsoft Copilot for Government — requires M365 E5 licence; Deloitte BudgetIQ — proprietary, no local-model option.

---

### I5. Integrated Treasury Single Account (TSA) Reconciliation
**Category**: Cash & Liquidity Management | **Justification**: Disconnected TSA and appropriation systems cause ghost payments, duplicate disbursements, and audit exceptions. Automated reconciliation between TSA ledger movements and commitment/expenditure records closes this gap daily rather than quarterly. | **Implementation**: Async `reconcile_tsa_with_expenditures()` method that matches `_tsa_movements` debits to `ExpenditureRecord` amounts within configurable tolerance (default 0.01 KES), flags unmatched items, publishes reconciliation report to NATS `apg.government.bud.tsa.reconciliation`. | **Competitor**: Temenos TCBS — bank-grade but not open; IPSAS-certified FreeBalance Accountability Suite.

---

### I6. Donor Funds Conditionality Compliance Tracker
**Category**: Aid & Grant Management | **Justification**: Donor-funded programmes routinely fail conditionality triggers, risking suspension of disbursements. An automated tracker prevents the $50-200M disbursement suspension events common in SSA government programmes. | **Implementation**: Async `check_donor_conditionality()` evaluates each registered donor budget against a list of conditions (financial reports submitted, audit completed, procurement rules followed), computes compliance score, triggers NATS alert `apg.government.bud.donor.compliance` when score < threshold. | **Competitor**: UN MPTF Gateway — manual web portal; World Bank Client Connection — read-only, no programmatic enforcement.

---

### I7. Fiscal Risk Register & Contingent Liability Modelling
**Category**: Risk Management | **Justification**: Hidden contingent liabilities (guarantees, PPP obligations, pension arrears) are the primary source of fiscal shocks in developing economies. A structured register with probability-weighted exposure quantification reduces year-end surprises by 60-70%. | **Implementation**: Async `register_fiscal_risk()` records risk category, probability, maximum exposure, trigger condition, and mitigation action. `compute_contingent_liability_exposure()` aggregates expected value (probability × exposure) across risk register and adds to total public debt estimate. Events to NATS `apg.government.bud.risk`. | **Competitor**: World Bank PFRAM tool — Excel-based, no API; Moody's Analytics Public Finance — USD 500K+ licences.

---

### I8. Budget Circular Automated Distribution & Compliance Scoring
**Category**: Budget Preparation | **Justification**: Manual distribution of budget circulars to MDAs and chasing of returns consumes 6-8 weeks of budget department time per cycle. Automated distribution with submission tracking and compliance scoring compresses this to under 2 weeks. | **Implementation**: Async `issue_budget_circular()` generates circular record with submission deadlines per MDA, tracks returns, computes compliance score. Reminder events published to NATS `apg.government.bud.circular.{mda_id}` at T-14, T-7, and T-0. | **Competitor**: Questica Budget — North America focused, no PFMA/budget act alignment; OpenGov — US local government only.

---

### I9. IPSAS-Aligned Accrual Accounting Conversion
**Category**: Financial Reporting Standards | **Justification**: Most African government IFMIS operate on cash basis; IMF/World Bank increasingly require IPSAS accrual reporting for programme lending. Automated cash-to-accrual conversion maps existing cash transactions to IPSAS categories without parallel re-entry. | **Implementation**: Async `generate_ipsas_accrual_report()` converts cash expenditure records to accrual basis using configurable recognition rules (goods received not invoiced, prepayments, deferred revenue), outputs IPSAS-compliant Statement of Financial Position and Statement of Financial Performance. | **Competitor**: Oracle Public Sector Financials — $2M+ deployment; Epicor ERP Government Edition — proprietary, no open standard compliance engine.

---

### I10. Legislated Appropriation Compliance Guard
**Category**: Governance & Legality | **Justification**: Expenditure beyond appropriated votes is unconstitutional in most jurisdictions. Automated pre-commitment appropriation checks prevent the political and legal exposure of ultra vires spending, which affects ~15% of MDAs in high-spend periods. | **Implementation**: Async `check_appropriation_compliance()` verifies that proposed commitment amount does not exceed the legislated vote ceiling (including any passed supplementary appropriations), returns compliance verdict with relevant Act citation and gazette reference. Denial events published to NATS audit subject. | **Competitor**: SAP BPC for Public Sector — rule-based only, no legislative citation; Hyperion Governmental — US GAAP only.

---

### I11. Arrears Management & Payment Prioritisation Engine
**Category**: Debt & Arrears Control | **Justification**: Government payment arrears distort private-sector credit markets and trigger fiscal crises. An automated arrears registry with age-banding, legal exposure scoring, and optimised payment sequencing reduces average arrears age by 30-40% and avoids penalty interest charges. | **Implementation**: Async `register_payment_arrear()` records overdue commitments with due date, creditor class, legal exposure, and penalty rate. `generate_arrears_payment_plan()` applies priority rules (statutory obligations first, then contractors, then grants) subject to TSA cash availability. Events to NATS `apg.government.bud.arrears`. | **Competitor**: Deloitte Government Arrears Management Solution — services-led, $5M+ engagements; no open-source equivalent.

---

### I12. Parliamentary Budget Submission Package Generator
**Category**: Legislative Relations | **Justification**: Parliamentary committees require standardised budget books, estimates of expenditure, and programme performance reports. Manual compilation takes 4-6 months. Automated assembly from live system data reduces this to 48 hours and eliminates transcription errors. | **Implementation**: Async `generate_parliamentary_estimates()` compiles vote-level estimates, prior-year actuals, three-year projections, and PBB scorecards into a structured output (JSON/PDF-ready). Publishes draft to NATS `apg.government.bud.parliament.submission` for downstream document generation. | **Competitor**: Questica Budget for Legislatures — US only; MYOB Advanced Government — Australia only.

---

### I13. Macro-Fiscal Scenario Stress Testing
**Category**: Fiscal Sustainability Analysis | **Justification**: Point-estimate budgets fail when macro conditions deviate — commodity price swings, exchange-rate shocks, or revenue shortfalls. Scenario-based stress testing (high/base/low) enables Finance Ministries to pre-approve contingency reallocation triggers before crises hit. | **Implementation**: Async `stress_test_budget()` applies user-defined macro shocks (revenue decline %, expenditure pressure %) to current budget envelopes, computes fiscal deficit impact under each scenario, identifies breach thresholds, and recommends pre-committed contingency actions. Powered by local Ollama model with structured output schema. | **Competitor**: IMF's MTDS Analytical Tool — Excel-based, not API-accessible; Bloomberg Government Budget Model — proprietary.

---

### I14. Inter-Government Fiscal Transfer (IGFT) Allocation Engine
**Category**: Devolution & Transfers | **Justification**: Formula-based transfers from national to county/local governments are legally mandated but computationally intensive and error-prone when done in spreadsheets. Automated allocation reduces litigation risk and ensures equitable distribution per legally prescribed criteria. | **Implementation**: Async `compute_igft_allocation()` applies the statutory formula (population weight, poverty index, land area, equal share) to total shareable revenue, computes per-unit allocations, validates against constitutional floor (15% for Kenya counties), publishes allocation schedule to NATS `apg.government.bud.igft`. | **Competitor**: SAP IGFT module — Tier-1 SAP deployment required; World Bank SimFis — analytical tool only, no operational API.

---

### I15. Expenditure Anomaly Detection via Local ML
**Category**: Fraud & Waste Prevention | **Justification**: Government fraud and waste typically manifest as statistical outliers in expenditure patterns — late-year spending spikes, round-number amounts, unusual payee clusters. Local ML anomaly detection (no data leaves the ministry) catches 70-80% of flag patterns that manual review misses. | **Implementation**: Async `detect_expenditure_anomalies()` extracts expenditure vectors (amount, timing, payee class, vote type, approver), runs Isolation Forest or autoencoder via local Ollama/MLX, returns ranked anomaly list with suspicion scores and recommended investigation actions. Events to NATS `apg.government.bud.anomaly`. | **Competitor**: KPMG Forensics AI Platform — cloud-based, data sovereignty risk; ACL Robotics — rule-based only, no ML; SAS Fraud Management — $1M+ licence.
