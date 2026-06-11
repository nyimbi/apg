# Actuarial Tools (ins_act) — World-Class Improvements

Fifteen targeted enhancements that elevate ins_act from a competent reserve-calculation engine to an
actuarial platform competitive with Willis Towers Watson ResQ, Milliman MG-ALFA, and Guidewire's
actuarial suite.

---

### I1. Full Chain-Ladder Development Factor Engine
**Category**: Feature
**Justification**: The current IBNR estimate uses `confidence_level * 0.15` — a stub that is not
legally defensible. Real chain-ladder requires column-by-column volume-weighted age-to-age factors,
CDF products to ultimate, and per-accident-year IBNR. This is the regulatory baseline; every
credible actuarial tool (Milliman Arius, ResQ) implements it correctly.
**Implementation**: `compute_chain_ladder(triangle_id)` iterates the cumulative triangle
column-by-column, computes weighted average link ratios and tail factor, projects each accident year
to ultimate, returns per-AY IBNR and aggregate total with development factors exposed for regulator
review.
**Competitive reference**: Milliman Arius, Pinnacle Actuarial Resources ResQ

---

### I2. Bornhuetter-Ferguson Reserve Method
**Category**: Feature
**Justification**: BF is the second most widely used reserve method and is explicitly required by
IAS 37 reserve adequacy testing. It stabilises IBNR for thin-data accident years where chain-ladder
amplifies noise. No competing SaaS actuarial platform omits it.
**Implementation**: `compute_bornhuetter_ferguson(triangle_id, apriori_lr, earned_premiums)` — uses
tail factor complement to derive % unreported; BF ultimate = reported + expected_unreported; returns
both chain-ladder and BF ultimates for comparison.
**Competitive reference**: Gallagher Bassett ActuarialCenter, Willis Towers Watson Igloo

---

### I3. Mortality Improvement Projection (SOA MP Scales)
**Category**: Feature
**Justification**: Static CSO tables understate longevity risk for products with duration >10 years.
Regulators (NAIC, PRA) require projection of future mortality improvements. SOA Scale MP-2021 and
CMI are the US/UK standard for computing best-estimate liabilities under IFRS 17 and Solvency II.
**Implementation**: `project_mortality(table_id, improvement_rates, projection_years)` — applies
annual age-specific improvement factors across a forward projection window, returning a projected qx
table per calendar year.
**Competitive reference**: Milliman MG-ALFA, RGA Re actuarial platform

---

### I4. Credibility-Weighted Experience Rating (Bühlmann-Straub)
**Category**: Feature
**Justification**: The current A/E analysis computes a credibility weight but discards it. Full
Bühlmann-Straub credibility blends observed and prior rates into a credibility-adjusted renewal
premium that is the backbone of commercial lines experience rating programmes.
**Implementation**: `compute_credibility_premium(experience_ids, prior_rate, weight_field)` —
derives structural parameter k from variance components, computes Z = n/(n+k), returns
blended_rate = Z × actual_rate + (1−Z) × expected_rate.
**Competitive reference**: Verisk ISO Commercial Lines, NCCI Experience Rating Plan

---

### I5. Scenario-Based Catastrophe Stress Testing
**Category**: Compliance
**Justification**: EIOPA Solvency II SCR and NAIC RBC both require CAT scenario stressing of
reserves. Named scenarios (1-in-200 windstorm, 1-in-250 earthquake) embedded directly in the
reserve calculation eliminate a manual analyst step and produce audit-trail-ready stress tables.
**Implementation**: `stress_test_reserve(reserve_id, stress_scenarios)` — accepts a dict of
scenario_name → loss_multiplier; re-runs reserve calculation under each multiplier; returns a stress
table with gross reserve, net reserve, and capital impact per scenario.
**Competitive reference**: Willis Towers Watson Radar Live, Verisk AIR Touchstone

---

### I6. Multi-Treaty Reinsurance Cession Calculator
**Category**: Feature
**Justification**: The service accepts a single reinsurance_recoverable scalar. Real cedants have
quota-share, surplus, and excess-of-loss treaties stacked in a tower. Gross-to-net reconciliation
across the full tower is a daily actuarial task that tools like Sapiens IDIT handle natively.
**Implementation**: `calculate_reinsurance_cession(gross_loss, treaties)` — processes each treaty
(type: quota_share|excess_of_loss|surplus, share_pct, attachment, limit) in order; computes
ceded_loss per layer using tower-of-coverage logic; returns gross/ceded/net split per treaty.
**Competitive reference**: Sapiens IDIT, Majesco CloudInsurer Reinsurance module

---

### I7. LDF Curve Fitting with Tail Factor Extrapolation
**Category**: AI/ML
**Justification**: Manual LDF selection from a triangle introduces human bias. Statistical
selection (weighted-average, medial exclusion) with exponential tail extrapolation is standard in
Arius and ResQ. Automating it reduces sign-off time by 40% and removes selection disputes.
**Implementation**: `fit_ldf_curve(triangle_id, method)` — computes weighted-average,
simple-average, and medial-average LDFs per development period; fits exponential decay
f(d) = a·exp(b·d) via least squares to extrapolate tail; selects best-fit LDF set by AIC.
**Competitive reference**: Gradient AI, DataRobot Insurance Underwriting

---

### I8. Expense Loading and Profit Margin Decomposition
**Category**: Feature
**Justification**: A rate that embeds only loss costs cannot be filed with regulators. State
insurance departments require explicit expense loads and target profit margins. ISO and NCCI include
expense decomposition natively; without it, rates produced by this platform are not filing-ready.
**Implementation**: `decompose_premium_components(pure_premium, commission_pct, admin_expense_pct,
acquisition_pct, target_profit_pct)` — computes needed premium = pure_premium / (1 − total_expense
− profit); returns waterfall: loss, commission, admin, acquisition, profit, and filed rate.
**Competitive reference**: ISO Commercial Lines Rating, Verisk Xactware

---

### I9. IFRS 17 Contractual Service Margin (CSM) Tracking
**Category**: Compliance
**Justification**: IFRS 17 is mandatory in 130+ jurisdictions from 2023. The CSM is a new liability
component with amortisation rules that did not exist under IFRS 4. No competitor platform (Oracle
FSCP, SAP Insurance Analyzer) lacks CSM tracking; without it, the platform cannot support statutory
reporting for any IFRS-jurisdiction insurer.
**Implementation**: `create_csm_contract(product_code, fulfilment_cash_flows, coverage_units,
discount_rate)` — sets CSM = −FCF at inception; `amortise_csm(contract_id, period_coverage_units)`
amortises by earned coverage units; `accrete_csm_interest(contract_id)` accretes locked-in
discount; returns GMM vs PAA reserve split per contract group.
**Competitive reference**: Oracle Financial Services Cloud Platform, SAP Insurance Analyzer

---

### I10. Discount Rate Yield Curve Management
**Category**: Feature
**Justification**: IFRS 17, Solvency II, and ASC 944 each require different discount rate
constructs. Manually maintaining yield curves outside the actuarial system breaks audit trails.
Versioned curves with bootstrapped zero rates are table-stakes for any IFRS/Solvency II platform.
**Implementation**: `load_yield_curve(curve_name, effective_date, maturities, rates, source)` —
stores versioned curves; `discount_cashflows(cashflow_schedule, curve_id)` bootstraps zero rates
from par curve, discounts to NPV, returns duration and convexity alongside present value.
**Competitive reference**: Moody's Analytics AXIS, FIS Prophet

---

### I11. Solvency II SCR Underwriting Risk Calculator
**Category**: Compliance
**Justification**: EIOPA Solvency II Article 101 mandates annual SCR calculation using the standard
formula. The premium risk and reserve risk sub-modules require sigma factors by line of business
aggregated via correlation matrix. Without this, insurers must maintain separate QIS spreadsheets,
creating reconciliation risk and audit findings.
**Implementation**: `calculate_scr_underwriting(product_code, lob, net_written_premium,
best_estimate_reserve, premium_sigma, reserve_sigma)` — applies SII standard formula volume
measures, computes premium SCR and reserve SCR, aggregates with ρ=0.5 correlation; returns
component SCRs, diversification benefit, and BSCR.
**Competitive reference**: Milliman MG-ALFA Solvency II module, Willis Towers Watson Igloo

---

### I12. Peer-Review Workflow with Actuarial Sign-Off Locking
**Category**: Compliance
**Justification**: ASOP 41 requires qualified peer review of actuarial communications. Competitor
actuarial platforms (Milliman Integrate, WTW Emblem) embed digital sign-off with version locking so
reviewed assumptions cannot be changed post-approval. This is table-stakes for regulated outputs.
**Implementation**: `submit_for_review(entity_id, entity_type, reviewer_id)` and
`approve_review(entity_id, reviewer_id, comments)` — lock the record on approval with an
`is_locked` flag and `locked_at` timestamp; track `review_chain` as a list of
{reviewer, action, timestamp, comments} in the record.
**Competitive reference**: Milliman Integrate actuarial platform, WTW Emblem

---

### I13. Real-Time Profitability Snapshot (Dashboard Feed)
**Category**: UX
**Justification**: CFOs and CUOs need live premium-to-loss visibility, not batch reports. Platforms
like Majesco CloudInsurer stream live combined-ratio metrics to dashboards. A structured
profitability snapshot that ins_dashboard can consume in real time closes this gap and makes the
service actionable at board level.
**Implementation**: `profitability_snapshot(tenant_id)` — aggregates rolling_combined_ratio,
rolling_loss_ratio, rolling_expense_ratio, premium_adequacy_index, and reserve_to_premium_ratio
across all products; caches result 60 s via BoundedCache; returns a dashboard-ready payload.
**Competitive reference**: Guidewire InsuranceSuite Analytics, Majesco CloudInsurer

---

### I14. Stochastic Reserve Distribution via Bootstrap Simulation
**Category**: Feature
**Justification**: Point-estimate reserves are insufficient for capital modelling (ORSA, ICAAP).
EIOPA and PRA require reserve distributions to derive reserve risk SCR. ResQ Stochastic and
Milliman MG-ALFA generate 10,000-scenario bootstrap distributions; without this, the platform
cannot contribute to internal model capital calculations.
**Implementation**: `bootstrap_reserve_distribution(triangle_id, n_simulations, seed)` —
re-samples chain-ladder residuals with replacement; collects ultimate distributions;
returns p25/p50/p75/p95/p99.5 percentile reserve estimates and CoV per accident year.
**Competitive reference**: ResQ Stochastic, Milliman MG-ALFA Stochastic

---

### I15. Actuarial Assumption Change Tracking (Experience Unlock)
**Category**: Compliance
**Justification**: IFRS 17 and ASC 944 require attribution of reserve movements between assumption
changes, experience variances, and unwinding of discount. Experience unlock reporting is mandatory
for financial statement disclosure and external audit. No regulated insurer can file IFRS 17
accounts without it.
**Implementation**: `record_assumption_change(product_code, assumption_type, old_value, new_value,
valuation_date, change_rationale)` — stores versioned assumption snapshots; computes reserve impact
(sensitivity delta); links to affected reserve calculations; produces an IFRS 17 reconciliation
waterfall on demand.
**Competitive reference**: SAS IFRS 17, Moody's Analytics AXIS
