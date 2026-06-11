# fin.dep — World-Class Improvements

Deposit Products Engine: 15 improvements to reach tier-1 banking software standards.

© 2025 Datacraft · Author: Nyimbi Odero

---

### I1. Promotional Rate Windows
**Category**: Product Configuration
**Justification**: Banks routinely offer introductory rates (e.g. "8% for first 3 months, then 6%") to acquire deposits. Without this, pricing ops must create duplicate products and migrate accounts manually — error-prone and operationally expensive.
**Implementation**: Add `PromotionalRate` sub-model (rate, start_date, end_date, revert_rate) to `InterestConfig`. `_resolve_effective_rate()` checks today against active windows before returning the tier rate.
**Competitor Reference**: Temenos T24 Product Builder promotional rate bands; Thought Machine Vault "smart contract" eligibility clauses.

---

### I2. Stepped / Escalating Interest Rate Schedules
**Category**: Interest Engine
**Justification**: Notice and call deposits in East Africa commonly escalate rates the longer principal stays (e.g. +0.25% per additional 30 days held). This drives retention and is standard in KCB, Equity, and Co-op product sheets.
**Implementation**: `ProductTerms.rate_steps: list[RateStep]` where `RateStep(days_held, delta_rate)`. `calculate_interest()` splits the period into segments, applying the delta at each threshold.
**Competitor Reference**: FNB South Africa Call Account stepped accrual; Stanbic Uganda Notice Deposit ladder.

---

### I3. Multi-Currency FX Conversion on Maturity
**Category**: Maturity Processing
**Justification**: Term deposits in KES, USD, EUR are common cross-border. At rollover/payout, banks convert principal + interest to a target currency using a treasury rate. Missing this forces manual FX journals and breaks GL reconciliation.
**Implementation**: `convert_maturity_to_currency(tenant_id, account_id, target_currency, fx_rate)` — applies FX, updates account currency, posts a GL conversion entry, records the rate used.
**Competitor Reference**: Oracle FLEXCUBE FX deposit conversion; Mambu multi-currency rollover.

---

### I4. Penalty Waiver Workflow with Approval Chain
**Category**: Operations / Compliance
**Justification**: Relationship managers routinely waive early-break penalties for VIP customers. Without an auditable waiver flow, waivers happen out-of-band (spreadsheets, emails), creating audit gaps that regulators flag.
**Implementation**: `request_penalty_waiver(tenant_id, account_id, reason, requested_by)` creates a `PenaltyWaiverRequest`; `approve_penalty_waiver(waiver_id, approved_by)` / `reject_penalty_waiver(waiver_id, rejected_by, reason)` complete the flow with full audit trail.
**Competitor Reference**: Temenos Workflow Engine penalty exception; Finacle relationship pricing exception module.

---

### I5. Interest Rate Scenario Comparison (Multi-Product Simulator)
**Category**: Analytics / Sales
**Justification**: Customer-facing advisors need to compare returns across products for a given principal and tenor in a single API call. Current `simulate_maturity()` is single-product only, forcing N sequential calls.
**Implementation**: `compare_products(tenant_id, principal, tenor_days, product_codes)` fans out `simulate_maturity()` across all requested products and returns a ranked `ComparisonResult` list sorted by net_interest descending.
**Competitor Reference**: Stanbic "Deposit Calculator" multi-product comparison; TymeBank savings rate comparison API.

---

### I6. Dormancy Detection and Fee Assessment
**Category**: Compliance / Fee Engine
**Justification**: Central Bank of Kenya and Bank of Uganda regulations require banks to classify accounts with no transaction activity for 12+ months as dormant and apply statutory dormancy fees or transfer balances to escrow. This is a compliance obligation, not a feature request.
**Implementation**: `classify_dormant_accounts(tenant_id, as_of_date, inactivity_days)` scans accounts, marks dormant ones, posts dormancy fee. `reactivate_account(tenant_id, account_id, reactivated_by)` reverses classification.
**Competitor Reference**: CBK Prudential Guideline CBK/PG/01 dormancy rules; Finacle Dormancy Management module.

---

### I7. Interest Capitalization Control (Capitalise vs. Pay-Out)
**Category**: Interest Engine
**Justification**: Some depositors prefer periodic interest credited to a linked current account rather than compounding. Banks need per-account override of the product-level capitalization setting to satisfy private banking clients.
**Implementation**: `set_interest_disposition(tenant_id, account_id, disposition, linked_payout_account)` where `disposition` is `CAPITALIZE | PAY_OUT`. `apply_interest()` checks this flag before crediting.
**Competitor Reference**: Temenos T24 AC-PARAM CAPINT field; Mambu interest settings per account.

---

### I8. Batch Maturity Sweep
**Category**: Operations
**Justification**: Banks have thousands of term deposits. Processing each maturity manually at EOD is operationally impossible. An automated sweep that handles all accounts reaching maturity on a given date is table-stakes for any core banking system.
**Implementation**: `batch_process_maturities(tenant_id, maturity_date)` finds all accounts with `maturity_date` <= target, applies the account's pre-set `MaturityInstruction` (defaulting to product `auto_rollover`), and returns a `BatchMaturityResult` with counts and error list.
**Competitor Reference**: Temenos COB maturity sweep job; Silverlake SIBS batch maturity run.

---

### I9. Accrual Reversal with GL Correction Entry
**Category**: Accounting Integrity
**Justification**: Month-end accruals sometimes need reversal due to rate corrections, backdated transactions, or system errors. Without a proper reversal mechanism, GL balances diverge from the interest ledger — a material audit finding.
**Implementation**: `reverse_accrual(tenant_id, account_id, accrual_date, reason, reversed_by)` creates a negating `AccrualEntry` with `is_reversal=True`, links it to the original, and generates a GL reversal stub.
**Competitor Reference**: Oracle FLEXCUBE interest reversal transaction IRVS; Temenos interest amendment.

---

### I10. Loyalty Bonus Rate on Deposit Renewal
**Category**: Retention / Pricing
**Justification**: Retaining maturing TD customers costs less than acquiring new ones. Loyalty bonuses (e.g. +0.5% on 2nd rollover, +1.0% on 3rd+) are a proven retention tool used by top-tier banks but require rollover-count tracking.
**Implementation**: Track `rollover_count` in the account record. `process_term_deposit_maturity()` with `ROLLOVER` increments count. `_resolve_tier_rate()` checks a product-level `loyalty_schedule: list[LoyaltyBonus]` to add the delta.
**Competitor Reference**: Stanbic Uganda "Flexi Fixed" loyalty rate; Standard Chartered KE renewal bonus.

---

### I11. Regulatory Concentration Limit Enforcement
**Category**: Risk / Compliance
**Justification**: Basel III and CBK regulations cap single-depositor concentration (e.g. no single depositor > 10% of total deposit base). Breaching this triggers regulatory reporting obligations. Real-time enforcement prevents the breach rather than detecting it after.
**Implementation**: `check_concentration_limit(tenant_id, account_id, new_deposit_amount, limit_pct)` aggregates all balances per depositor ID, computes concentration, raises `ConcentrationLimitError` if the new deposit would breach the configured threshold.
**Competitor Reference**: Temenos Limit Server concentration check; Finastra Fusion Capital concentration monitoring.

---

### I12. Interest Rate Floor and Cap Guards
**Category**: Risk / Interest Engine
**Justification**: Floating-rate products indexed to CBR or LIBOR replacements (SOFR, SONIA) need floor/cap guardrails so that extreme central bank moves do not produce negative interest or rates exceeding regulatory caps. These are contractual obligations in floating-rate deposit agreements.
**Implementation**: `InterestConfig.rate_floor: Decimal = 0` and `rate_cap: Decimal | None`. `_resolve_effective_rate()` clamps the resolved rate: `max(floor, min(cap, rate))`.
**Competitor Reference**: Temenos floating rate with floor/cap; Thought Machine Vault interest floor directive.

---

### I13. Statement Generation (Account Statement Export)
**Category**: Customer Service / Reporting
**Justification**: Deposit account holders require periodic statements showing opening balance, all interest postings, fees debited, and closing balance. This is a regulatory requirement (CBK Banking Act s.24) and a customer expectation.
**Implementation**: `generate_account_statement(tenant_id, account_id, from_date, to_date)` aggregates postings, fee records, and accrual entries into a `AccountStatement` model with line items and running balance.
**Competitor Reference**: Finacle Account Statement module; FLEXCUBE Customer Account Statement (CAS).

---

### I14. Product Cloning with Override Support
**Category**: Product Management
**Justification**: Product managers routinely create variants of existing products (e.g. "Premium Savings" as a clone of "Classic Savings" with higher rate). Without clone support, they must re-enter all fields — a slow, error-prone process that produces configuration drift.
**Implementation**: `clone_product(tenant_id, source_code, new_code, new_name, overrides)` deep-copies the source product, applies field overrides, persists under new code, and initialises rate history with a "cloned_from" entry.
**Competitor Reference**: Temenos Product Cloning (TAFJ); Mambu "duplicate product" action.

---

### I15. Real-Time Effective Annual Yield (EAY) Computation
**Category**: Analytics / Transparency
**Justification**: Regulators (CBK, CMA) and sophisticated depositors require disclosure of Effective Annual Yield — the true return accounting for compounding and tax — not just the nominal rate. Standardised EAY disclosure is mandatory under Kenya's Finance Act 2023.
**Implementation**: `get_effective_annual_yield(tenant_id, product_code, principal, tax_rate_override)` computes `EAY = ((1 + r/n)^n - 1) × (1 - wht_rate/100)` for compound products and `r × (1 - wht_rate/100)` for simple, returning a `YieldResult` with gross_eay, net_eay, and disclosure_text.
**Competitor Reference**: CMA Kenya disclosure requirement; European MiFID II KIID yield figure.
