# SACCO Lending — World-Class Improvements

Benchmarked against Mambu, Temenos Microfinance, CRAFT Silicon, Musoni System, and Finflux.

---

### I1. Dynamic Loan Restructuring
**Category:** Loan Lifecycle Management
**Justification:** Borrowers facing genuine hardship need restructuring (tenor extension, rate reduction, payment holiday) without write-off. Musoni System offers one-click restructuring with full audit trail and schedule regeneration. Without this, SACCOs lose members to competitors and inflate write-off ratios unnecessarily.
**Implementation:** `restructure_loan()` — accepts new tenor, new rate, holiday months, reason, and approver; regenerates the amortisation schedule from the current outstanding balance; preserves original terms in `restructuring_history` list; transitions status back to `active`.
**Competitor Reference:** Musoni System "Loan Restructuring" module; Mambu "Loan Rescheduling" API.

---

### I2. Penalty & Late-Fee Engine
**Category:** Arrears Monetisation
**Justification:** Flat-rate daily penalties (typically 0.1–0.5% of overdue amount) are the primary SACCO deterrent against chronic arrears. Temenos TCMF auto-accrues penalties on overdue installments. Without automated penalty accrual, officers manually compute penalties, introducing errors and revenue leakage.
**Implementation:** `accrue_penalties()` — iterates active arrears records, computes `penalty = overdue_amount × daily_penalty_rate × arrears_days`, stores as a separate `penalty_ledger` dict, caps penalty at product-level `max_penalty_pct`, updates `loan["accrued_penalty"]`.
**Competitor Reference:** Temenos TCMF "Penalty Charge" workflow; CRAFT Silicon Lending penalty engine.

---

### I3. Loan Top-Up (Enhancement)
**Category:** Customer Retention
**Justification:** Members with good repayment records frequently need incremental capital. Requiring a full new application is friction. Finflux offers top-ups where the net disbursement = requested amount − outstanding balance, with a blended rate. This retains members and reduces churn to banks.
**Implementation:** `apply_loan_topup()` — validates existing active loan, checks repayment record percentage ≥ threshold, creates a new loan linked via `parent_loan_id` with `loan_type = "topup"`, nets outstanding balance against new amount, regenerates schedule.
**Competitor Reference:** Finflux "Loan Top-Up"; Mambu "Refinance Loan" API.

---

### I4. Flat-Rate Interest Schedule
**Category:** Interest Calculation
**Justification:** Many Kenyan SACCOs still use flat-rate interest (interest computed on original principal throughout). The current service only implements reducing balance. CRAFT Silicon supports both, allowing product managers to choose per product. Offering flat-rate enables SACCO compliance with legacy product terms.
**Implementation:** `_build_flat_rate_schedule()` — `monthly_interest = principal × annual_rate / 1200`; each installment = `(principal / term_months) + monthly_interest`; no balance reduction on interest calculation. Invoked by `approve_loan` when `product["interest_method"] == "flat_rate"`.
**Competitor Reference:** CRAFT Silicon Lending; Kenya SACCO Societies Regulatory Authority (SASRA) guidelines.

---

### I5. Savings-Linked Collateral Lock
**Category:** Risk Mitigation
**Justification:** SACCO loans are typically secured partly by member savings. When a loan is approved, a portion of savings should be frozen as collateral until the loan is repaid. Temenos TCMF supports automated lien placement on savings accounts. Without this, a member can withdraw savings while in default, removing the SACCO's primary security.
**Implementation:** `place_savings_lien()` — records a lien record linking `loan_id`, `member_id`, `lien_amount`, and `savings_account_id`; transitions lien status through `placed → released` on loan closure; `release_savings_lien()` triggers on `loan_closed` event.
**Competitor Reference:** Temenos TCMF "Account Lien"; Musoni "Savings-Linked Loan Security".

---

### I6. Group/Solidarity Loan Support
**Category:** Product Breadth
**Justification:** Group lending (Grameen-style) is the dominant SACCO model in rural Kenya. Each group member is jointly liable. Finflux and Musoni both support group loans with individual sub-schedules and joint liability matrices. Without group lending, the service is unusable for a large segment of Kenyan SACCOs.
**Implementation:** `create_group_loan()` — accepts a list of member shares, creates individual sub-loans linked to a parent group loan record, applies joint-liability flag so any member default triggers arrears on all sub-loans; `get_group_loan_summary()` aggregates by group.
**Competitor Reference:** Finflux "Group Loans"; Musoni "Group Lending" module.

---

### I7. Automated Repayment Allocation Waterfall
**Category:** Repayment Accuracy
**Justification:** When a repayment is received, it should be allocated in a defined order: penalties first, then interest, then principal. Ad-hoc allocation causes interest accrual errors. Mambu enforces a configurable waterfall (penalty → fee → interest → principal). The current implementation applies the full payment directly to outstanding balance without waterfall logic.
**Implementation:** `_allocate_repayment()` — private method that takes `payment_amount` and loan state; allocates sequentially to `accrued_penalty`, then `accrued_interest`, then `outstanding_principal`; returns a breakdown dict stored on each repayment record as `allocation_breakdown`.
**Competitor Reference:** Mambu "Repayment Allocation Method"; Temenos TCMF payment waterfall.

---

### I8. SASRA Regulatory Reporting
**Category:** Compliance
**Justification:** Kenya's SACCO Societies Regulatory Authority (SASRA) requires quarterly returns: PAR30, PAR90, write-off ratios, provisioning rates. Mambu Kenya and CRAFT Silicon generate SASRA-formatted reports automatically. Without this, compliance officers export raw data and build reports manually — a regulatory risk.
**Implementation:** `generate_sasra_report()` — computes PAR30, PAR60, PAR90, write-off ratio, provisioning requirement per SASRA Prudential Guidelines 2020, returns structured report with `period_end_date` and `sacco_registration_number`; stores in `regulatory_reports` dict.
**Competitor Reference:** Mambu Kenya SASRA module; CRAFT Silicon regulatory pack.

---

### I9. Loan Insurance Claim Processing
**Category:** Risk Transfer
**Justification:** Most SACCO loans carry mandatory credit life insurance. When a borrower dies or is permanently disabled, an insurance claim should clear the outstanding balance. Temenos TCMF and Musoni both have insurance claim workflows. Without this, deceased member estates remain on the arrears ledger indefinitely.
**Implementation:** `submit_insurance_claim()` — validates loan status, creates claim record with `claim_type` (`death | disability | redundancy`), supporting document references, claim amount equal to outstanding balance, transitions loan to `insurance_claim_pending`; `settle_insurance_claim()` zeroes balance and closes loan on confirmed settlement.
**Competitor Reference:** Temenos TCMF "Insurance Claim"; Musoni "Credit Life Insurance" module.

---

### I10. Bullet / Balloon Loan Schedules
**Category:** Product Breadth
**Justification:** Asset finance and mortgage products frequently use bullet repayment structures where only interest is paid monthly and the full principal is due at maturity. Finflux and Mambu support bullet schedules as a first-class schedule type. SACCOs offering vehicle or land loans need this urgently.
**Implementation:** `_build_bullet_schedule()` — generates `term_months - 1` interest-only installments, then a final installment of `principal + last_month_interest`; invoked when product `repayment_type == "bullet"`. Adds `repayment_type` field to product schema.
**Competitor Reference:** Finflux "Bullet Loan"; Mambu "Interest Only" repayment method.

---

### I11. Loan Officer Performance Dashboard
**Category:** Operational Intelligence
**Justification:** Loan officers are evaluated on disbursement volume, collection rate, PAR contribution, and approval throughput. CRAFT Silicon and Finflux both expose officer-level KPIs. Without officer-level attribution, managers cannot identify underperformers or reward top collectors.
**Implementation:** `loan_officer_metrics()` — groups loans by `approved_by` / `disbursed_by`; computes per-officer: disbursement count, total volume, average term, collection rate, PAR contribution; returns ranked list with a composite performance score.
**Competitor Reference:** CRAFT Silicon "Field Officer Reports"; Finflux "Collection Agent Dashboard".

---

### I12. Early Repayment / Prepayment Penalty
**Category:** Revenue Protection
**Justification:** Early repayment on flat-rate loans causes interest income loss. SACCOs typically charge a prepayment penalty of 1–3% on the remaining balance. Temenos TCMF enforces configurable prepayment penalty per product. Without this, early full repayment silently erodes forecast interest income.
**Implementation:** `compute_early_settlement_quote()` — accepts `settlement_date`, computes remaining schedule interest, applies `early_settlement_fee_pct` from product config to outstanding balance, returns a settlement quote with expiry (48 hrs); `settle_loan_early()` records fee and closes loan.
**Competitor Reference:** Temenos TCMF "Early Repayment Charge"; Mambu "Prepayment" configuration.

---

### I13. Multi-Tier Approval Workflow
**Category:** Governance & Controls
**Justification:** Large loans (e.g., > KES 500K) require committee approval rather than single-officer sign-off. Musoni and Mambu implement configurable approval matrices (amount thresholds → required approver roles). Without tiered approvals, SACCOs fail internal audit on large loan authorisation.
**Implementation:** `escalate_loan_for_approval()` — checks loan amount against product `approval_tiers` list (list of `{max_amount, required_role}`); creates an `ApprovalRequest` record per tier; `record_tier_approval()` stores each approver's sign-off; loan transitions to `approved` only when all required tiers are satisfied.
**Competitor Reference:** Musoni "Loan Approval Workflow"; Mambu "Multi-Level Authorisation".

---

### I14. Rollover / Renewal Automation
**Category:** Customer Lifecycle
**Justification:** Short-term emergency and school-fees loans are routinely renewed at maturity. Manual renewals create processing delays. Finflux automates rollover: a new loan is created referencing the old one, with the closing balance optionally carried forward. This reduces officer workload by 40% on high-volume products.
**Implementation:** `renew_loan()` — verifies loan is `closed` or within 30 days of maturity; creates a new loan linked via `renewed_from_loan_id`; optionally carries forward an unpaid balance as the opening principal of the new loan; preserves member guarantor list from original loan.
**Competitor Reference:** Finflux "Loan Renewal"; CRAFT Silicon "Loan Rollover".

---

### I15. Dynamic Provisioning Engine
**Category:** Financial Reporting
**Justification:** SASRA Prudential Guidelines 2020 mandate specific loan-loss provisioning: 1% (current), 3% (1–30 days), 20% (31–90 days), 50% (91–180 days), 100% (>180 days). Temenos TCMF computes provisions nightly. Without automated provisioning, SACCOs understate credit losses and risk regulatory sanctions.
**Implementation:** `compute_loan_loss_provision()` — iterates all active/arrears loans, classifies by arrears bucket per SASRA schedule, applies provision rate, sums to portfolio-level provision requirement; stores per-loan provision in `provision_ledger`; returns summary with `total_provision_required`, `provision_coverage_ratio`, and bucket breakdown.
**Competitor Reference:** Temenos TCMF "Loan Loss Provisioning"; Mambu "Provisioning Rules" engine.
