# SASRA Regulatory Reporting — World-Class Improvement Roadmap

© 2025 Datacraft · Author: Nyimbi Odero

---

### I1. Stress-Test Simulation Engine
**Category**: Risk Analytics
**Justification**: SASRA's own supervisory framework calls for SACCOs to demonstrate forward-looking capital adequacy. Without shock scenarios, a SACCO cannot anticipate the impact of a 20% NPL spike before it hits the statutory breach level. This gap is universally cited in SASRA examination findings.
**Implementation**: `stress_test_capital_adequacy(scenarios: list[StressScenario])` — apply parameterised shocks (loan book deterioration %, deposit withdrawal %, haircut on govt securities) to a ledger snapshot, recompute all ratios, and return delta tables showing which scenario triggers the first breach and what the capital headroom is.
**Competitor Reference**: Kenya Commercial Bank's ICAAP (Internal Capital Adequacy Assessment Process) documents use exactly this multi-scenario stress framework; SASRA's own Prudential Guidelines §4.4 explicitly require it for large deposit-taking SACCOs.

---

### I2. Trend Analysis & Ratio Trajectory Forecasting
**Category**: Predictive Compliance
**Justification**: Static point-in-time ratios are the minimum requirement. Leading SACCOs and their auditors need to answer "where will our CAR be in 6 months?" before they file. No current implementation provides rolling-window analysis or linear-regression forecasting against historical ledger snapshots.
**Implementation**: `analyse_ratio_trends(tenant_id, ratio_name, lookback_quarters)` — pull historical snapshots, compute ratios at each period, fit a linear trend, project N quarters forward, flag if the trajectory crosses a threshold before the next filing date.
**Competitor Reference**: Mambu's regulatory analytics module includes ratio trajectory cards; Kenya's Equity Bank ALCO dashboards project CAR using 4-quarter rolling regression.

---

### I3. Corrective Action Plan (CAP) Generator
**Category**: Regulatory Workflow
**Justification**: When a SACCO breaches a ratio, SASRA requires a written Corrective Action Plan within 30 days (SACCO Societies Act Cap 490B, §35). Manually drafting these is error-prone. A structured CAP with quantified targets, timelines, and board sign-off checklist dramatically reduces regulatory risk.
**Implementation**: `generate_corrective_action_plan(tenant_id, as_of_date)` — detect all breaches, for each breach compute the exact capital/liquidity amount needed to restore compliance, generate structured action items (member capital call, loan recovery targets, asset liquidation schedule), set 30-day and 60-day review milestones, and return a machine-readable plan with an exportable PDF-ready dict.
**Competitor Reference**: CU*Answers (US credit union core) auto-generates NCUA Prompt Corrective Action plans; SASRA examination reports repeatedly cite SACCOs' failure to have written CAPs.

---

### I4. Statutory Reserve Adequacy Monitor
**Category**: Capital Management
**Justification**: The SACCO Societies Act requires transfer of at least 10% of annual net surplus to a statutory reserve until the reserve equals the minimum core capital threshold (Reg 22). Many SACCOs under-fund the reserve while reporting compliant CARs. Current service does not track this obligation independently.
**Implementation**: `check_statutory_reserve_adequacy(tenant_id, year)` — extract net surplus from income statement, compute required annual transfer (10%), compare cumulative statutory reserve to minimum threshold, flag whether the FY appropriation is compliant, and compute the top-up needed.
**Competitor Reference**: Craft Silicon's Bankers Realm SACCO module includes a statutory reserve tracker as a standard compliance widget; Bank of Ghana credit union supervisors use an identical statutory reserve monitor.

---

### I5. Dividend Restriction Enforcer
**Category**: Member Protection / Regulatory Gate
**Justification**: SASRA regulations prohibit dividend payments when key ratios are breached. Without a programmatic gate, a SACCO's finance department may inadvertently approve dividends during a compliance breach, triggering further regulatory sanction. This is one of the most common examination findings.
**Implementation**: `check_dividend_eligibility(tenant_id, proposed_dividend_amount, as_of_date)` — verify all SASRA ratio compliance, check that the proposed dividend does not breach the post-distribution CAR or liquidity floor, return an eligibility decision with per-ratio breakdown and a SASRA-language rationale string.
**Competitor Reference**: Temenos T24 includes a dividend gate in its regulatory compliance module; the Reserve Bank of Zimbabwe's microfinance guidelines use the same dividend restriction framework.

---

### I6. Multi-Period Peer Benchmarking
**Category**: Comparative Analytics
**Justification**: A SACCO CEO's most important question is "how do our ratios compare to similar SACCOs?" SASRA publishes aggregate sector statistics quarterly. Embedding a peer-group benchmark provides context that a standalone compliance check cannot — a 12% CAR looks weak if the sector median is 18%.
**Implementation**: `benchmark_against_sector(tenant_id, sector_data: dict, as_of_date)` — accept a sector statistics payload (median, p25, p75 per ratio from SASRA Supervision Report), compute the SACCO's percentile position for each ratio, flag outlier ratios, and return a structured benchmarking summary.
**Competitor Reference**: NCUA's Call Report system provides automatic peer benchmarking for all US credit unions; SASRA's annual Supervision Report contains the sector-level data that feeds this.

---

### I7. Liquidity Stress Buffer Calculator
**Category**: Liquidity Risk
**Justification**: The 15% liquidity minimum is a static floor. SASRA's prudential guidelines also reference a "Net Stable Funding Ratio" concept for large SACCOs. Computing the actual stressed liquidity position (after a 3-day deposit run scenario) is required for any SACCO seeking external lines of credit from banks.
**Implementation**: `calculate_liquidity_stress_buffer(tenant_id, run_rate_pct, as_of_date)` — model a N-day deposit withdrawal at run_rate_pct, compute residual liquid assets after the run, determine how many days of coverage remain, and report whether the SACCO passes a survival-horizon test (minimum 5 days).
**Competitor Reference**: Basel III LCR methodology adapted for SACCOs; Kenya Bankers Association's liquidity framework uses identical 3/5-day survival horizon tests.

---

### I8. Regulatory Filing Reminder & Penalty Estimator
**Category**: Compliance Operations
**Justification**: SASRA imposes penalty fees for late filings. The penalty schedule (SASRA Fee Regulations, 2015) is: KES 2,000/day for quarterly returns, KES 5,000/day for annual accounts. No current tooling computes the accruing penalty so the board can quantify the cost of delay.
**Implementation**: `estimate_late_filing_penalty(tenant_id, return_type, filing_date)` — compute days overdue, apply the applicable SASRA penalty rate, return total accrued penalty, remaining-days cost, and whether the 90-day suspension-trigger threshold has been crossed.
**Competitor Reference**: FSRA (Ontario) credit union regulation engine includes automatic penalty accrual; Bank of Uganda MFI supervision system has a daily penalty ticker.

---

### I9. Loan Write-Off Recommendation Engine
**Category**: Asset Quality
**Justification**: Loss-band loans (>365 DPD, 100% provisioned) that remain on the book inflate gross portfolio, overstate NPL ratios, and mask the true quality of the performing book. SASRA's own examination reports consistently identify failure to write off fully-provisioned loans as a reporting weakness.
**Implementation**: `recommend_loan_writeoffs(tenant_id, as_of_date, min_dpd)` — identify all loss-band loans meeting write-off criteria (100% provisioned, DPD > min_dpd), compute post-write-off ratios, verify no ratio breach is triggered by the write-off, and return a write-off schedule with board resolution template text.
**Competitor Reference**: Temenos Loan IQ includes an automated write-off recommendation module; Kenya's CBA (now NCBA) uses quarterly automated write-off sweeps with regulator pre-notification.

---

### I10. Regulatory Ratio Sensitivity Analysis
**Category**: Capital Planning
**Justification**: Management needs to answer "how much new lending can we do without breaching our LDR?" or "how large a deposit can we take before our liquidity ratio drops below 15%?". Marginal sensitivity tables answer these questions directly and underpin the SACCO's business plan.
**Implementation**: `calculate_ratio_sensitivity(tenant_id, as_of_date, ratio_name, delta_range)` — sweep the primary driver of the ratio (e.g. gross loans for LDR, liquid assets for liquidity) across a range, recompute the ratio at each step, identify the headroom until breach, and return a sensitivity table.
**Competitor Reference**: Oliver Wyman's ICAAP toolkit for banks includes identical sensitivity sweep tables; any SASRA-compliant business plan now requires ratio sensitivity analysis per SASRA Supervision Guidance Note 3.

---

### I11. Cross-Ratio Conflict Detector
**Category**: Regulatory Quality Assurance
**Justification**: A SACCO can simultaneously report a compliant CAR and a breached LDR, which is internally consistent but triggers different supervisory escalation paths. Some ratio combinations are arithmetically inconsistent (e.g. liquidity ratio implying negative borrowings). Automated cross-validation catches data-entry errors before submission.
**Implementation**: `validate_ratio_consistency(tenant_id, as_of_date)` — run a set of cross-ratio invariant checks (balance sheet identity, LDR/CAR directional consistency, liquidity vs CAR interaction), flag arithmetic inconsistencies with specific field references and suggested corrections.
**Competitor Reference**: KPMG's regulatory filing review checklist for Kenyan SACCOs includes 14 cross-ratio consistency checks; SASRA's portal validator enforces 6 of these on submission.

---

### I12. SASRA Examination Readiness Score
**Category**: Supervisory Readiness
**Justification**: SASRA conducts annual on-site examinations scored against a published framework (governance, capital, asset quality, management, earnings, liquidity — CAMEL). A readiness score lets a SACCO self-assess and prioritise remediation before the examiner arrives, reducing the risk of a poor rating.
**Implementation**: `calculate_examination_readiness_score(tenant_id, as_of_date)` — score each CAMEL dimension using available data (ratios for C, A, E, L; governance flags for M), compute a composite score on a 1-5 SASRA scale, identify the weakest component, and suggest the three highest-impact improvements.
**Competitor Reference**: NCUA's CAMEL rating system for US credit unions is the direct model; SASRA's examination manual (2019) uses an identical CAMEL structure.

---

### I13. Capital Injection Planning Tool
**Category**: Capital Management
**Justification**: When a SACCO's CAR is below the 10% minimum, the board needs a quantified capital injection plan: how much new share capital is needed from members, over what period, to restore compliance. Simply reporting the shortfall is insufficient — the regulator expects a funded recovery plan.
**Implementation**: `plan_capital_injection(tenant_id, as_of_date, target_car_pct, injection_months)` — compute current CAR shortfall, model monthly member share capital calls of equal size to reach target_car_pct within injection_months, verify the injection pace is realistic given member base size, return an amortisation schedule of capital injections with expected CAR at each step.
**Competitor Reference**: WOCCU (World Council of Credit Unions) capital restoration tools use identical injection modelling; Kenya's Unaitas SACCO used a multi-year share capital mobilisation plan post-SASRA intervention in 2018.

---

### I14. Consolidated Group Reporting
**Category**: Multi-Entity Compliance
**Justification**: Several large SACCOs operate subsidiary entities (insurance fronts, property companies). SASRA increasingly examines consolidated exposures. Single-entity reporting misses group-level concentration risks and can mask exposure to affiliates that inflate the parent SACCO's apparent capital.
**Implementation**: `generate_consolidated_report(primary_tenant_id, subsidiary_tenant_ids: list[str], as_of_date)` — consolidate balance sheets (eliminating intra-group balances), recompute group-level ratios, identify where subsidiary exposures consume parent capital, and produce a consolidated compliance status with entity-level drill-down.
**Competitor Reference**: KPMG's Kenya banking supervision toolkit includes group consolidation for credit union holding structures; the East African Community's cross-border SACCO supervision framework mandates consolidated reporting for SACCOs with operations in multiple EAC countries.

---

### I15. Automated Audit Trail & Evidence Package
**Category**: Governance / Audit
**Justification**: SASRA examiners require SACCOs to produce a full audit trail for every ratio reported in a quarterly return: the source data, the calculation steps, the reviewer sign-off, and the date. Manual compilation takes 2–3 days. An automated evidence package cuts this to minutes and reduces the risk of inconsistency between the return and supporting schedules.
**Implementation**: `generate_audit_evidence_package(tenant_id, year, quarter)` — for each ratio in the quarterly return, emit a structured evidence record containing: input field values with source keys, calculation formula applied, result, SASRA threshold tested, compliance outcome, and a hash of the ledger snapshot for immutability. Bundle all records into a signed JSON manifest exportable as a ZIP.
**Competitor Reference**: Wolters Kluwer OneSumX generates automated regulatory evidence packages for Basel III submissions; KPMG Kenya's SACCO audit methodology requires identical per-ratio evidence files.
