# SACCO Group Lending — World-Class Improvements

© 2025 Datacraft — Author: Nyimbi Odero

---

### I1. Interest-Bearing Group Loans with Amortisation Schedule
**Category:** Core Lending

**Justification:** Current loans track outstanding balance but have no interest model. Real SACCO group loans carry fixed or declining-balance interest. Without interest computation, the SACCO cannot determine monthly instalments, total cost of credit, or interest income — making financial reporting meaningless.

**Implementation:** Add `interest_rate_annual_pct` and `interest_method` (FLAT / REDUCING) to `GroupLoan`. Add `build_amortisation_schedule` method that returns monthly instalment amounts, principal split, interest split, and cumulative balance for each period. Store schedule on the loan object. `record_group_repayment` should allocate payment between interest and principal using the schedule.

**Competitor Reference:** M-Shwari group loans, Faulu Kenya SACCO — both publish full amortisation schedules to borrowers.

---

### I2. Loan-to-Savings Ratio Enforcement
**Category:** Credit Risk

**Justification:** All production SACCOs enforce a multiplier cap (e.g. a member may borrow at most 3× their accumulated savings). Skipping this check allows credit to be issued without collateralisation, violating prudential norms and creating unsecured group exposure.

**Implementation:** Add `max_loan_to_savings_multiplier` field to `Group`. In `apply_group_loan`, compute each member's savings share vs. their requested disbursement and raise `LoanToSavingsExceeded` if any member would breach the multiplier. Surface per-member LTS ratios in the loan application response.

**Competitor Reference:** Stima SACCO, Co-operative Bank SACCO — enforce 3× or 4× LTS hard caps in their loan origination rules.

---

### I3. Penalty Accrual for Late Repayments
**Category:** Arrears Management

**Justification:** Late payments without financial consequence remove the incentive for timely repayment. Competitors charge a fixed daily or monthly late-payment fee that accumulates until cleared. Without it, the arrears figure understates the true cost of default.

**Implementation:** Add `penalty_rate_daily_pct` and `penalty_balance` to `GroupLoan`. Add `accrue_loan_penalties` method that calculates days overdue since each missed instalment and posts the penalty amount to `penalty_balance`. `record_group_repayment` should allocate incoming funds: penalties first, then interest, then principal (waterfall).

**Competitor Reference:** Equity Bank Chama Loan product — 5% p.m. penalty on overdue amounts.

---

### I4. Group Dividend / Interest-on-Savings Distribution
**Category:** Savings Management

**Justification:** Investment clubs and welfare groups must distribute end-of-year dividends to members proportional to their average savings balance throughout the year. This is a legal obligation for registered SACCOs (SACCO Societies Act, Kenya) and a core member benefit.

**Implementation:** Add `distribute_group_dividend` method. Calculate each member's time-weighted average balance over the dividend period. Compute individual dividend = (member avg balance / group avg balance) × total dividend pool. Post dividend lines to member accounts and emit a `dividend_distributed` audit event.

**Competitor Reference:** Kenya Union of Savings & Credit Co-operatives (KUSCCO) annual dividend distribution model; Harambee SACCO dividend computation engine.

---

### I5. Emergency / Welfare Loan Sub-Facility
**Category:** Product Diversity

**Justification:** Welfare groups routinely issue small emergency loans from the pooled savings fund — not from an external lender. Current model only handles external borrowing. A welfare disbursement from the group's own savings pool with a short repayment tenure is a distinct product with different risk and accounting treatment.

**Implementation:** Add `issue_welfare_loan` method. Checks group type is `WELFARE` or that an emergency fund balance exists. Deducts from `group_savings_balance`, creates a sub-loan record linked to a single member (not the whole group). Tracks repayment back into the fund.

**Competitor Reference:** Faulu Kenya — welfare fund loans capped at 1× monthly contribution, 3-month tenor, disbursed same day.

---

### I6. Meeting Attendance & Quorum Tracking
**Category:** Governance

**Justification:** Registered Chamas under Kenya's Chama Bill require quorum (≥50% attendance) to pass resolutions (loan approvals, member exits). Without attendance records the SACCO has no audit trail proving valid consent for financial decisions.

**Implementation:** Add `record_meeting_attendance` method. Store an `AttendanceRecord` with `present_member_ids`, `apology_member_ids`, and `quorum_achieved: bool`. Link attendance records to `GroupContribution` (contribution sessions are usually held at meetings). Surface quorum status in `approve_group_loan` — warn if the approval meeting lacked quorum.

**Competitor Reference:** M-Changa platform records digital attendance at virtual Chama meetings; Kenya Cooperative Societies Audit standards require meeting minute trails.

---

### I7. SMS/WhatsApp Contribution Reminders (Event Emission)
**Category:** Member Engagement

**Justification:** Contribution compliance rates drop sharply without reminders. The capability should emit structured notification events 3 days before each expected contribution date so a downstream messaging layer can send M-PESA prompts, SMS, or WhatsApp nudges. This is a primary driver of compliance improvement in East Africa.

**Implementation:** Add `generate_contribution_reminders` method. Compute each member's next expected contribution date from `meeting_frequency` and last contribution date. Emit `contribution_reminder` events with `member_id`, `group_id`, `expected_date`, `expected_amount`, and `days_until_due`. Return a list of pending reminders for the caller to dispatch.

**Competitor Reference:** M-Changa, Savings Circle (Tanda) — both auto-generate push notifications 3 days and 1 day before contribution due dates.

---

### I8. Group Credit Score Export for Individual Member Loans
**Category:** Credit Infrastructure

**Justification:** A member's group repayment behaviour is the strongest predictor of their individual creditworthiness in markets without formal credit bureaux. Exporting a group-informed credit signal for individual members enables cross-capability underwriting in the broader APG platform.

**Implementation:** Add `get_member_credit_signal` method. For a given member, aggregate: contribution compliance rate, fraction of group loan repayment contributed on time, any joint-liability defaults, and time in group. Return a `MemberCreditSignal` dict suitable for consumption by the individual loan underwriting capability (`fintech_sacco_loan`).

**Competitor Reference:** Pezesha (Kenya) — uses Chama repayment history as primary credit variable for individual MSME loans; Branch International similar.

---

### I9. Merry-Go-Round Cycle Reset & Multi-Cycle Tracking
**Category:** Merry-Go-Round

**Justification:** Once all members have received their kitty, the MGR group starts a new cycle. Current code marks `merry_go_round_received = True` permanently with no way to start cycle 2. Groups that have been running for years cannot continue without manual intervention.

**Implementation:** Add `reset_merry_go_round_cycle` method. Archives the current cycle (cycle number, all round records), resets `merry_go_round_received = False` for all active members, increments `cycle_number` on the group, optionally resets the rotation order. Return a cycle summary including total distributed in the completed cycle.

**Competitor Reference:** Tanda (formerly Savings Circle) — explicitly models multi-cycle ROSCA groups with cycle history dashboards.

---

### I10. Group Loan Restructuring
**Category:** Arrears Recovery

**Justification:** When a group is in persistent arrears, SACCOs restructure rather than write off: extend the tenure, capitalise overdue interest, or reduce the instalment. Without a restructure method the only options are write-off (loss) or continued arrears accumulation — neither is good practice.

**Implementation:** Add `restructure_group_loan` method. Parameters: `new_tenure_months`, `capitalise_arrears: bool`, `new_interest_rate: Decimal | None`. Creates a restructure event on the loan, adjusts `outstanding_balance` if arrears capitalised, recalculates amortisation schedule, transitions status from `ARREARS` back to `ACTIVE`. Emit `loan_restructured` audit event.

**Competitor Reference:** KCB SACCO, Equity SACCO — publish formal loan restructure policies with up to 2 restructure events per loan.

---

### I11. Bulk Group Loan Write-Off
**Category:** Credit Portfolio Management

**Justification:** SACCO regulators require provisioning and eventual write-off of non-performing loans (NPLs) after 360 days in arrears. Without a write-off mechanism, bad loans persist on the books indefinitely, overstating assets and understating credit losses.

**Implementation:** Add `write_off_group_loan` method. Validates that `days_in_arrears >= write_off_threshold_days` (configurable, default 360). Transitions loan status to `WRITTEN_OFF`, records the write-off amount, date, and authorising officer. Optionally reduces per-member balances to zero. Emit `group_loan_written_off` audit event.

**Competitor Reference:** CBK Prudential Guidelines (Kenya) — mandate write-off after 365 days in arrears; Stima SACCO implements automated write-off workflows.

---

### I12. Contribution Projection & Savings Target Tracking
**Category:** Financial Planning

**Justification:** Groups set annual savings targets (e.g. "accumulate KES 500,000 by December"). Without a projection engine, members cannot see whether their current contribution pace will meet the target, and group officers cannot adjust contribution amounts proactively.

**Implementation:** Add `project_group_savings` method. Inputs: `target_amount`, `target_date`. Calculates current trajectory using average monthly contribution rate. Returns `months_to_target`, `required_monthly_contribution` to hit target on time, `projected_shortfall_or_surplus`, and a `projection_series` list of (date, projected_balance) tuples.

**Competitor Reference:** Acorns, Savings Circle — both provide timeline projection graphs; Standard Chartered Kenya's Savings Goal feature.

---

### I13. Multi-Tier Guarantor System
**Category:** Credit Enhancement

**Justification:** For larger group loans, individual member guarantorship by external parties materially reduces default risk. Some groups require that each member bring an external guarantor before the group can access a loan. This is standard practice in MFI (microfinance institution) group lending globally.

**Implementation:** Add `add_loan_guarantor` and `get_loan_guarantors` methods. `GroupLoanGuarantor` model stores `member_id`, `guarantor_external_id`, `guaranteed_amount`, `guarantor_type` (PERSONAL / PROPERTY / SALARY_DEDUCTION), and `confirmed: bool`. `approve_group_loan` can enforce a minimum guarantor coverage ratio before approval proceeds.

**Competitor Reference:** Kenya Women Finance Trust (KWFT) — requires each member to present two guarantors before group loan disbursement; SMEP Microfinance Bank.

---

### I14. Automated Group Performance Benchmarking
**Category:** Analytics

**Justification:** A group's absolute performance score is only meaningful when compared to peer groups (same type, similar size, same SACCO branch). Benchmarking reveals whether a group is outperforming or lagging its cohort, enabling targeted officer intervention.

**Implementation:** Add `benchmark_group_performance` method. Computes performance scores for all groups of the same `group_type` and `meeting_frequency` within the tenant. Returns the target group's percentile rank, the cohort median score, best-in-cohort group id, and delta to median. Cache benchmarks with a TTL to avoid full recomputation on every call.

**Competitor Reference:** M-Pesa Chama analytics dashboard; Tala's portfolio benchmarking for group lending agents.

---

### I15. Audit Trail Export with Tamper-Evidence Hash Chain
**Category:** Compliance & Governance

**Justification:** Regulators (CBK, Saccos Societies Regulatory Authority — SASRA) require that group financial records be tamper-evident and exportable for audit. A hash chain where each event references the hash of the previous event makes retroactive data falsification computationally detectable.

**Implementation:** Add `export_audit_trail` method. Each audit event is stored with a `sha256` hash = `sha256(previous_hash + event_json)`. Genesis hash is a known constant. Export returns events with their hashes and a `chain_valid: bool` flag computed by re-verifying all hashes from genesis. Output can be JSON or CSV.

**Competitor Reference:** Chainalysis-style ledger integrity; CBK Prudential Reporting guidelines on audit trail requirements; Temenos SACCO module includes immutable ledger export.
