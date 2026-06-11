# LMS World-Class Improvements

**Capability**: fin_lms — Loan Management System
**Author**: Nyimbi Odero — Datacraft © 2025
**Date**: 2026-06-11

---

### I1. Partial Prepayment with Configurable Waterfall Override
**Category**: Core Repayment Engine
**Justification**: Borrowers frequently make ad-hoc partial prepayments mid-cycle. Banks like KCB and Equity allow the customer (or RM) to specify whether surplus funds reduce tenor or reduce instalment. Without this, all prepayments default to reducing tenor, which may misalign with customer expectations and create regulatory mis-statements.
**Implementation**: Add `prepay_with_options(tenant_id, loan_id, amount, date, strategy: Literal["reduce_tenor","reduce_instalment","custom_waterfall"])`. When `reduce_instalment`: recalculate PMT for remaining tenor. When `reduce_tenor`: advance future principal amortisation. Post a dedicated GL line (DR Cash / CR Loans Receivable) and re-emit a revised schedule.
**Competitor Reference**: Mambu `prepayment_strategy` enum; Temenos Transact `PAYMENT.DETAIL` override field.

---

### I2. Interest Accrual Engine (Daily Accrual, Month-End Posting)
**Category**: Accounting Accuracy / IFRS 9
**Justification**: IFRS 9 requires daily interest accrual on the effective interest rate basis. Current implementation only applies interest at installment due dates. This understates interest income between posting dates and makes P&L lumpy — a deficiency flagged in CBK on-site examinations.
**Implementation**: Add `accrue_daily_interest(tenant_id, as_of_date)` batch job. Compute `balance × EIR / 365` per loan per day since last accrual. Post DR Accrued Interest Receivable (1210) / CR Interest Income (4100). Store last accrual date on the loan. Month-end `capitalise_accrued_interest(tenant_id, period_end)` moves from accrued to earned.
**Competitor Reference**: Finastra Fusion Loan IQ daily accrual engine; Flexcube `AC_BASIS` daily accrual batch.

---

### I3. Effective Interest Rate (EIR) / XIRR Calculation
**Category**: IFRS 9 Compliance / Analytics
**Justification**: IFRS 9 amortised cost measurement mandates EIR calculation that incorporates origination fees, transaction costs, and discount. Reporting the contractual rate instead of EIR is a material audit finding. The EIR feeds both income recognition and the ECL (Expected Credit Loss) discount rate.
**Implementation**: Add `calculate_eir(loan_id, origination_fees, transaction_costs)`. Use Newton-Raphson on the IRR equation over the full cashflow stream (disbursement as negative, all installments + fees as positive). Return `{eir: Decimal, xirr: Decimal, total_cost_of_credit: Decimal}`. Cache on loan record.
**Competitor Reference**: SAP Bank Analyzer `EIR_CALC`; nCino EIR amortization feature; Moody's RiskCalc.

---

### I4. Expected Credit Loss (ECL) — IFRS 9 Stage Bucketing
**Category**: IFRS 9 / Regulatory Capital
**Justification**: CBK's IFRS 9 guidance (2019) requires all regulated institutions to bucket loans into Stage 1/2/3 and compute 12-month vs. lifetime ECL. Mapping DPD-based CBK classification to IFRS 9 stages is mandatory for financial statements. Current provision model only reflects CBK prudential rates, not IFRS 9 ECL.
**Implementation**: Add `compute_ecl_provision(tenant_id, loan_id, pd: Decimal, lgd: Decimal, ead: Decimal, stage: int)`. Stage 1 → 12-month ECL. Stages 2/3 → lifetime ECL (sum over remaining cashflows discounted at EIR). Post incremental GL entry for ECL movement. Expose `batch_compute_ecl(tenant_id, as_of_date)`.
**Competitor Reference**: Oracle FSDF ECL engine; Moody's Scenario Analyzer; Wolters Kluwer OneSumX IFRS 9 module.

---

### I5. Covenant Monitoring and Breach Alerts
**Category**: Risk Management / Relationship Banking
**Justification**: Term loans and corporate facilities carry financial covenants (DSCR, leverage ratio, current ratio). Breach of a covenant is a default event requiring accelerated provisioning or calling the loan. Without automated monitoring, covenants are checked manually at best quarterly, allowing undetected breaches. Standard in any mid-market lending system.
**Implementation**: Add `register_covenant(tenant_id, loan_id, covenant_type, threshold, breach_action)` and `check_covenant_compliance(tenant_id, loan_id, current_value, as_of_date)`. On breach: emit event, trigger demand notice escalation, optionally freeze drawdowns, set `LoanStatus.COVENANT_BREACH`. Store covenant history for audit.
**Competitor Reference**: Finastra Loan IQ covenant tracking; Salesforce Financial Services Cloud covenant monitoring; nCino covenant management module.

---

### I6. Instalment-Level Partial Pay Tracking (Split Installment Clearing)
**Category**: Repayment Accuracy
**Justification**: When a borrower pays less than a full instalment, the current service marks the instalment as `partial` but does not enforce FIFO ordering within the instalment itself (interest before principal within a single instalment). This causes incorrect GL entries — interest income may be understated when partial payments are received.
**Implementation**: Add `get_detailed_arrears_breakdown(tenant_id, loan_id)` returning per-instalment due/paid/outstanding split: `{installment_no, due_interest, paid_interest, due_principal, paid_principal, overdue_days, status}`. Enhance waterfall to enforce strict FIFO within each instalment (interest then principal per instalment before moving to next).
**Competitor Reference**: Temenos WealthSuite instalment split; Mambu installment-level tracking.

---

### I7. Loan Top-Up (Additional Drawdown on Existing Facility)
**Category**: Product Flexibility
**Justification**: Revolving credit and overdraft products allow borrowers to draw additional amounts within an approved limit after partial repayment. Mobile lending (e.g. KCB M-Pesa, Fuliza) is built on this pattern. Without top-up support the LMS forces a new loan origination, breaking the customer relationship continuity and creating duplicate GL entries.
**Implementation**: Add `topup_loan(tenant_id, loan_id, additional_amount, topup_date, approved_by)`. Validate `additional_amount <= (approved_limit - outstanding_balance)`. Add additional principal to outstanding balance, recalculate schedule from current date, post GL disbursement entry. Emit top-up event for audit trail.
**Competitor Reference**: Mambu `add_disbursement`; Temenos `LOAN.PAYMENT` INCREASE action; Flexcube additional disbursement.

---

### I8. Collateral Tracking and Forced Sale Value
**Category**: Risk Management / Recovery
**Justification**: CBK prudential guidelines require net-of-collateral provisioning for secured loans. Without LGD-adjusted provisions, the bank over-provisions on secured NPLs (inflating loss expense) or under-provisions on unsecured NPLs (capital inadequacy risk). Tracking collateral FSV (Forced Sale Value) enables accurate provision netting.
**Implementation**: Add `register_collateral(tenant_id, loan_id, collateral_type, fsv, market_value, haircut_rate, valuation_date)` and `get_collateral_coverage(tenant_id, loan_id)`. Integrate with `calculate_required_provision` to reduce provision by eligible collateral: `max(0, outstanding - net_collateral_value) × provision_rate`. Revalue collateral annually via `revalue_collateral(...)`.
**Competitor Reference**: Mambu collateral module; Flexcube ELCM collateral management; SAP Collateral Management.

---

### I9. Collections Workflow Automation (Escalation Ladder)
**Category**: Collections Operations
**Justification**: Effective loan collections requires a deterministic escalation ladder: SMS reminder → email reminder → phone → formal demand → legal → referral. Current implementation sends a single notice type on demand. Without automated escalation, collectors forget steps, regulatory safe-harbour timelines are breached, and courts reject recovery proceedings citing improper notice.
**Implementation**: Add `run_collections_escalation(tenant_id, as_of_date)` batch. Define `CollectionsPolicy` (configurable per product): e.g. DPD 5 → SMS; DPD 15 → email; DPD 30 → formal letter; DPD 60 → legal; DPD 90 → write-off recommendation. Escalate each loan to next stage, post notice, update `collections_stage` on loan. Idempotent — skip loans already at or past the stage.
**Competitor Reference**: Temenos Collections; Nucleus FinnOne collections module; FICO Debt Manager.

---

### I10. Fee Schedule Engine (Disbursement, Processing, Annual, Exit Fees)
**Category**: Revenue Accounting
**Justification**: Real lending products carry a fee stack: origination (1–2%), processing (flat KES), annual facility fee, exit/prepayment fee. Current model treats fees as a single `total_fees` aggregate with no structure. IFRS 9 requires origination fees to be amortised over loan life (integral to EIR), not expensed at disbursement. Without a fee schedule, income is mis-stated in year 1.
**Implementation**: Add `apply_fee(tenant_id, loan_id, fee_type: FeeType, amount, due_date)` and `amortise_fees(tenant_id, loan_id, as_of_date)`. `FeeType` enum: `ORIGINATION, PROCESSING, ANNUAL_FACILITY, EXIT, INSURANCE`. IFRS 9-qualifying fees (ORIGINATION) are deferred via contra-asset and amortised to interest income over tenor. Non-qualifying fees are expensed immediately.
**Competitor Reference**: Finastra Loan IQ fee engine; nCino fee management; Flexcube SC (Service Charges).

---

### I11. Bulk Portfolio Operations with Idempotency Keys
**Category**: Scalability / Operations
**Justification**: Nightly batch jobs (arrears run, accrual, provision update) currently process loans sequentially in-memory and have no crash recovery. In production, a batch of 50,000 loans that crashes at loan 30,000 leaves the portfolio in a half-updated state. This is operationally dangerous and causes GL imbalances that require manual correction.
**Implementation**: Add `batch_run(tenant_id, as_of_date, job_type: BatchJobType, idempotency_key: str)`. Store job state in `BatchJob` record: `{id, tenant_id, job_type, as_of_date, idempotency_key, started_at, completed_at, processed, errors, status}`. On re-run with same key, skip already-processed loans using a `processed_loan_ids` set. Emit completion event for monitoring.
**Competitor Reference**: Mambu batch jobs with idempotency; Temenos `BATCH.JOB.CONTROL`; AWS DynamoDB idempotency patterns.

---

### I12. Loan Participations / Syndication Splits
**Category**: Capital Markets / Corporate Banking
**Justification**: Large corporate and infrastructure loans are routinely syndicated across multiple banks. The lead arranger must track each participant's share of cashflows (principal, interest, fees) and pass-through payments. Without syndication support, lead banks use spreadsheets, creating reconciliation errors, delayed pass-throughs, and regulatory reporting failures.
**Implementation**: Add `register_participation(tenant_id, loan_id, participant_id, share_pct, commitment_amount)` and `allocate_payment_to_participants(tenant_id, loan_id, repayment_id)`. On repayment, calculate each participant's pro-rata share and emit `ParticipantPayment` events for downstream settlement. Track `lead_bank_fee` as a separate income line.
**Competitor Reference**: Finastra Loan IQ syndications; Broadridge LoanServ; WSO by Alter Domus.

---

### I13. Regulatory Reporting Pack (CBK, CRB, IFRS 9 Disclosures)
**Category**: Compliance / Regulatory
**Justification**: CBK requires monthly statutory returns (Form CBK-LR1 loan book, Form CBK-PC provisions) and quarterly CRB submissions (TransUnion, Metropol). Generating these from raw loan data requires complex SQL aggregations that differ per regulatory body. Centralising this in the LMS prevents inconsistencies between reports and reduces compliance cost.
**Implementation**: Add `generate_cbk_loan_register(tenant_id, reporting_date)`, `generate_cbk_provision_return(tenant_id, reporting_date)`, and `generate_crb_submissions(tenant_id, period)`. Each returns a structured dict matching the regulator's schema and a CSV/JSON export. Include data quality checks (null checks, balance reconciliation) before output.
**Competitor Reference**: Oracle Financial Services Regulatory Reporting; Wolters Kluwer AxiomSL; FIS Metavante regulatory module.

---

### I14. Multi-Currency Loan Support with FX Revaluation
**Category**: International Banking / FX Risk
**Justification**: USD and EUR denominated loans are common for trade finance, infrastructure, and corporate clients. FX-denominated loans require monthly revaluation (translate to KES at spot rate), with P&L impact posted to the FX translation reserve. Without this, the balance sheet and P&L misstate foreign currency exposures — a core CBK and IFRS requirement.
**Implementation**: Add `revalue_fx_loan(tenant_id, loan_id, spot_rate, revaluation_date)`. Compute `kes_equivalent = outstanding_balance_ccy × spot_rate`. Delta vs. prior revaluation → post FX gain/loss GL entry (DR/CR FX Translation Reserve 3100; opposite side Loans Receivable translated balance 1200). Store `kes_equivalent` and `last_revaluation_rate` on the loan.
**Competitor Reference**: Temenos T24 FX revaluation batch; Flexcube FX revaluation; Finastra Fusion `FX.REVALUATION`.

---

### I15. Loan Scoring Integration and Automatic Risk-Band Repricing
**Category**: Dynamic Risk Pricing
**Justification**: Credit risk changes over the life of a loan. A borrower who was B+ at origination may deteriorate to C- after a restructure or adverse financial event. Static rate-setting means the bank under-prices risk on deteriorated borrowers and over-prices on improved ones. Automated risk-band repricing at review dates aligns interest income with actual risk, is required by risk-based pricing frameworks, and is standard in digital lending platforms.
**Implementation**: Add `trigger_risk_review(tenant_id, loan_id, new_score: Decimal, review_date, scoring_model: str)`. Map score to rate band via a configurable `RiskBandTable` (score_min, score_max, rate_adjustment_bps). If new rate differs from current rate by >= `reprice_threshold_bps`, auto-call `reprice_loan(...)`. Emit `RiskReviewEvent` for audit. Notify borrower 30 days before effective date.
**Competitor Reference**: Kabbage/American Express dynamic pricing engine; Lendio risk-band repricing; Mambu pricing engine plugin.
