# SACCO Member Registry — World-Class Improvements

---

### I1. Risk-Tiered Member Scoring
**Category:** Credit Risk / Analytics
**Justification:** Every serious SACCO (M-Shwari, Stanbic Kenya, Equity Zimele) scores members on a composite risk profile before loan exposure. Pure share-capital data alone is insufficient for guarantor vetting and loan risk decisions.
**Implementation:** Calculate a 0–100 score from share capital, tenure, KYC completeness, outstanding guarantees, and payment history. Persist score version and recompute on every material event (share purchase, guarantor activity, fee payment). Expose `get_member_risk_score()` and `recalculate_all_risk_scores()`.
**Competitor Reference:** Equity Bank's EazzyBanking scores members at onboarding and monthly; M-Shwari uses a 12-factor credit score before any lending.

---

### I2. Guarantor Exposure Limit Enforcement
**Category:** Credit Risk / Compliance
**Justification:** Kenyan SACCO Societies Regulatory Authority (SASRA) guidelines require that a guarantor's total exposure (sum of guaranteed amounts across all active loans) not exceed 3× their share capital. Without enforcement, a member can guarantor unlimited loans leading to systematic default cascades.
**Implementation:** Before creating any guarantor relationship, sum all active `utilized_amount` values for that guarantor and compare against `max_exposure = share_capital × EXPOSURE_MULTIPLIER` (configurable, default 3.0). Raise `GuarantorExposureExceededError` with current exposure figures included in the exception payload.
**Competitor Reference:** Stima DT SACCO Kenya enforces this exactly. SASRA Prudential Guidelines (Regulation 22) mandates it.

---

### I3. Member Dividend Calculation Engine
**Category:** Financial Operations
**Justification:** Dividend declaration is the primary retention mechanism for SACCO members. Calculating pro-rata dividends based on weighted average share holdings over a fiscal period is complex but essential. Most SACCOs do this manually in spreadsheets — a first-class service method delivers enormous value.
**Implementation:** `calculate_dividends(tenant_id, fiscal_year, dividend_rate_percent)` iterates share transaction history per member, computes time-weighted average shares held, applies rate, and returns a dividend schedule. Persists dividend records; emits `dividends_calculated` event.
**Competitor Reference:** Mwalimu National SACCO publishes annual dividend schedules; Nairobi University SACCO automates pro-rata calculations in their core banking system.

---

### I4. KYC Document Expiry Tracking and Re-verification Alerts
**Category:** Compliance / Regulatory
**Justification:** Kenyan ID documents expire (passports: 10 years; driving licences: 5 years). SASRA AML/CFT guidelines require SACCOs to flag members with expired KYC documents and freeze loan access until re-verified. Without expiry tracking, regulatory exposure is unbounded.
**Implementation:** Add `document_expiry_date` to KYC records. `check_kyc_expiry(tenant_id)` returns all members whose KYC document expires within a configurable look-ahead window (default 90 days). `flag_expired_kyc(member_id)` transitions member to `kyc_status = "expired"` and `status = "restricted"`.
**Competitor Reference:** KCB Group KYC lifecycle management; Cooperative Bank of Kenya re-KYC mandated every 3 years.

---

### I5. Beneficiary / Next-of-Kin Share Inheritance Workflow
**Category:** Member Lifecycle / Estate
**Justification:** Upon member death, share capital must be transferred to registered next-of-kin under legal authority (probate letter, succession cert). Without a structured workflow, the SACCO faces legal exposure and member-estate disputes. Death is one of the four EXIT_REASONS already in the system but has no dedicated inheritance path.
**Implementation:** `initiate_inheritance(deceased_member_id, beneficiary_national_id, legal_doc_ref, processed_by)` validates next-of-kin match against member record, creates an inheritance record, parks shares in escrow state. `complete_inheritance(inheritance_id, approved_by, settlement_ref)` transfers share capital to beneficiary member (creating one if absent) and marks the exit complete.
**Competitor Reference:** Kenya National Police SACCO estate transfer protocol; Stanbic Kenya succession claim workflow.

---

### I6. Bulk KYC Import via CSV/JSON Batch
**Category:** Operational Efficiency
**Justification:** Large SACCOs onboarding employees of a single employer (e.g., county government, hospital) receive bulk member lists from the employer's HR system. Manual one-by-one creation is operationally impossible at scale. Safaricom SACCO and Police SACCO both do batch onboarding.
**Implementation:** `bulk_create_members(records: list[dict], created_by, tenant_id)` validates each record, deduplicates by national_id within batch and against existing members, creates all valid records in one pass. Returns `{created: int, failed: int, results: list, errors: list[{row, reason}]}`.
**Competitor Reference:** Safaricom SACCO bulk payroll-deduction onboarding; Co-op Bank bulk employer onboarding portal.

---

### I7. Share Withdrawal / Partial Redemption
**Category:** Financial Operations
**Justification:** SASRA allows share withdrawal with board approval after minimum holding period. Without this, the service models shares as one-way (purchase/transfer only), which is factually incorrect and blocks legitimate member financial operations.
**Implementation:** `withdraw_shares(member_id, shares, withdrawal_reason, approved_by, payment_ref, tenant_id)` checks minimum holding period (configurable), ensures remaining shares >= minimum_shares, reduces share_capital proportionally. Emits `shares_withdrawn` event. Records in share_transactions with type `sacco_share_withdrawal`.
**Competitor Reference:** Kenya Police SACCO partial share redemption; Ufundi SACCO shares withdrawal policy (SASRA-compliant, 25% max per year).

---

### I8. Member Financial Health Dashboard Aggregation
**Category:** Analytics / Member Service
**Justification:** A member interacting with a SACCO app needs a single aggregated view: shares, active guarantees (exposure), entry fees paid, and outstanding obligations. Building this as a service method avoids N+1 database queries in the presentation layer and enables caching.
**Implementation:** `get_member_financial_health(member_id, tenant_id)` aggregates: share capital, total shares, active guarantee count and total exposure, fees paid (by type), calculated risk score, loan eligibility estimate (share_capital × max_loan_multiplier). Returns a single structured dict designed for dashboard rendering.
**Competitor Reference:** M-Shwari app dashboard; Equity Zimele member portal summary card.

---

### I9. Member Merge (Duplicate Resolution)
**Category:** Data Quality / Operations
**Justification:** Bulk imports and data migrations routinely create duplicate member records. Without a merge facility, duplicates accumulate, creating double share capital, split transaction histories, and ambiguous guarantor graphs. This is a known pain point for every SACCO using multi-channel onboarding.
**Implementation:** `merge_members(primary_id, duplicate_id, merged_by, tenant_id)` validates both records belong to the same tenant and are not both active (one must be pending or suspended). Reassigns all share transactions, KYC records, fees, and guarantor relationships from duplicate to primary. Marks duplicate as `status="merged"` with `merged_into` pointer.
**Competitor Reference:** Salesforce Financial Services Cloud dedup; Mambu core banking member merge utility.

---

### I10. Configurable Member Tier / Segment Classification
**Category:** Product / Segmentation
**Justification:** SACCOs offer differentiated products (loan limits, interest rates, dividend rates) based on member tier (Bronze/Silver/Gold/Platinum by share capital range). Without tier logic in the registry, downstream loan and dividend services must duplicate tier rules.
**Implementation:** `classify_member_tier(member_id, tier_config: dict, tenant_id)` evaluates share_capital against configurable thresholds. `auto_reclassify_all(tenant_id, tier_config)` bulk reclassifies. Tier stored on member record with effective date. Emits `member_tier_changed` when tier changes.
**Competitor Reference:** Kenya Commercial Bank tiered SACCO product suite; Nairobi Water SACCO Premier/Classic/Standard tiers.

---

### I11. Dormancy Detection and Reactivation Workflow
**Category:** Member Lifecycle / Compliance
**Justification:** SASRA defines a dormant account as one with no transaction for 12+ months. Dormant members require specific handling: notification, dormancy fee levy, and eventual write-off process. Without automated detection, SACCOs violate SASRA reporting requirements.
**Implementation:** `detect_dormant_members(tenant_id, dormancy_threshold_days=365)` compares last transaction date (max of last share purchase, last fee, or activation date) against threshold. `flag_dormant(member_id, flagged_by, tenant_id)` transitions to `status="dormant"`, emits event. `reactivate_from_dormancy(member_id, reactivated_by, tenant_id)` reverses with reactivation fee logic.
**Competitor Reference:** SASRA Prudential Guidelines Regulation 18 (dormancy); Standard Chartered Kenya dormant account policy.

---

### I12. Employer-Linked Payroll Deduction Tracking
**Category:** Financial Operations / Employer Integration
**Justification:** Most formal-sector SACCOs collect share contributions and loan repayments via employer payroll deductions (check-off system). Without tracking the employer link and deduction schedule, reconciliation against payroll remittances is manual and error-prone.
**Implementation:** Add `employer_code`, `employee_number`, `payroll_deduction_amount`, `deduction_frequency` to member record (via `update_member_payroll_details`). `reconcile_payroll_remittance(tenant_id, employer_code, period, remittance_ref, total_amount, line_items)` matches line items to members, posts contributions, flags discrepancies.
**Competitor Reference:** Teachers Service Commission (TSC) SACCO check-off; Stanbic Kenya employer remittance portal.

---

### I13. Real-Time Duplicate National ID Detection Across Tenants
**Category:** Fraud Prevention / Data Integrity
**Justification:** In multi-SACCO deployments (platform model), the same individual may attempt to register across multiple SACCOs using slight name variations but the same national ID. Cross-tenant duplicate detection (with privacy controls — only flag, never expose data) is critical for platform operators.
**Implementation:** Maintain a platform-level `_national_id_registry: dict[str, set[str]]` (national_id → set of tenant_ids). On `create_member`, check if national_id already registered in another tenant and emit a `cross_tenant_duplicate_detected` alert (configurable: warn-only vs block). Expose `get_cross_tenant_member_report(national_id, requesting_tenant)` returning count (not detail) of other tenants.
**Competitor Reference:** iPay Africa fraud detection; GSMA Mobile Identity cross-operator duplicate detection.

---

### I14. Member Communication Event Log
**Category:** Compliance / Audit / CRM
**Justification:** SASRA requires SACCOs to maintain records of communications sent to members (suspension notices, dividend notifications, KYC rejection letters, exit confirmation). Without a communication log tied to member records, auditors find compliance gaps.
**Implementation:** `log_member_communication(member_id, comm_type, channel, reference, content_summary, sent_by, tenant_id)` persists a communication record. `list_member_communications(member_id, tenant_id, comm_type=None)` retrieves history. Communication types: `kyc_rejection`, `activation_notice`, `suspension_notice`, `dividend_notice`, `exit_confirmation`, `dormancy_warning`.
**Competitor Reference:** Salesforce Financial Services Cloud member communication log; Temenos Transact CRM audit trail.

---

### I15. Share Capital Minimum Adequacy Check Before Guarantor Assignment
**Category:** Credit Risk / Business Rules
**Justification:** A guarantor with trivially small share capital (e.g., 1 share × KES 100 = KES 100) should not be allowed to guarantee a loan of KES 500,000. Without a minimum adequacy check at guarantor relationship creation, the guarantor network provides false security. This is distinct from I2 (exposure limit) — this is a floor check, not a ceiling check.
**Implementation:** `create_guarantor_relationship` gains `minimum_guarantor_share_capital` parameter (default: configurable platform constant). Raises `InsufficientGuarantorShareCapitalError` with current and required amounts if guarantor's `share_capital < minimum_guarantee_amount`. Expose `validate_guarantor_eligibility(guarantor_id, guarantee_amount, tenant_id)` as a standalone check callable by the loans service.
**Competitor Reference:** Stima DT SACCO minimum share capital requirement for guarantors; SASRA loan policy guidelines.
