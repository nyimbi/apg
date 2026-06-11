# Loan Management System — User Guide

**Version**: 1.1.0 | **Platform**: APG `fin` domain  
**Author**: Nyimbi Odero — Datacraft © 2025

---

## 1. Introduction

The Loan Management System (LMS) manages the full post-origination lifecycle of
every loan from the moment it is disbursed until final closure. It integrates
with `fintech_lending` (origination) and `fin/gl` (general ledger) and supports
CBK (Central Bank of Kenya) and Basel II prudential guidelines out of the box.

### What LMS does

- Disburses loans and generates amortisation schedules (5 methods)
- Records repayments using a configurable waterfall (penalties → fees → interest → principal)
- Tracks arrears and classifies loans per CBK DPD thresholds
- Calculates and posts required provisions per CBK matrix (CBK) and ECL (IFRS 9)
- Handles restructuring, moratoriums, write-offs, and recoveries
- Produces portfolio quality reports (NPL, PAR>30, PAR>90, provision coverage)
- Sends demand notices and manages collections referrals
- Supports early settlement with interest rebate calculation
- Computes Effective Interest Rate (EIR/XIRR) for IFRS 9 amortised cost
- Runs IFRS 9 ECL stage bucketing (Stage 1 / 2 / 3)
- Handles loan top-ups and additional drawdowns on existing facilities
- Tracks collateral with FSV and haircut for net-of-collateral provisioning
- Runs automated collections escalation ladder (configurable DPD policy)
- Applies structured fee schedule with IFRS 9 deferral support
- Generates CBK regulatory reports (CBK-LR1 loan register)
- Revalues foreign-currency loans to base currency with FX P&L posting
- Posts daily IFRS 9 interest accrual entries

---

## 2. Loan Lifecycle

```
PENDING_DISBURSEMENT
      │ disburse_loan()
      ▼
   ACTIVE ──────────────────────────── record_repayment() ──► balance=0 → CLOSED
      │ days_past_due > 0
      ▼
  IN_ARREARS
      │ days_past_due ≥ 90
      ▼
     NPA
      │ approve write-off
      ▼
  WRITTEN_OFF ──► record_recovery() ──► RECOVERED
      │
      │ grant_moratorium()
      ▼
  MORATORIUM ──► schedule regenerated, status returns to ACTIVE
      │
      │ restructure_loan()
      ▼
  RESTRUCTURED ──► new schedule, ACTIVE resumes
```

---

## 3. Amortisation Methods

### 3.1 Reducing Balance

Equal principal instalments; interest calculated on declining balance.
Best for: retail term loans, agricultural loans.

```
Month 1:  Principal = P/n,  Interest = Balance × r/12
Month 2:  Principal = P/n,  Interest = (Balance - P/n) × r/12
...
```

### 3.2 Flat Rate

Interest computed on original principal for every period.
Total interest = P × r × tenor.  Higher effective rate than advertised.

### 3.3 French Annuity (PMT)

Constant total payment every period.  Early periods: high interest, low principal.
Later periods: low interest, high principal.  Standard mortgage method.

```
PMT = P × r(1+r)^n / ((1+r)^n - 1)
```

### 3.4 Bullet

Interest paid monthly; entire principal due at maturity.
Best for: working capital, overdraft facilities.

### 3.5 Interest Only

Identical to Bullet. Interest is the periodic payment; principal balloon at end.

---

## 4. Repayment Waterfall

Every payment is allocated in strict priority order:

| Priority | Component | Rationale |
|----------|-----------|-----------|
| 1 | Penalties | Deterrent; must be cleared first |
| 2 | Fees | Bank charges |
| 3 | Interest | Earned income |
| 4 | Principal | Reduces exposure |

Any unallocated balance is held as "float" pending further payments.

---

## 5. Arrears & NPA Classification

LMS runs `calculate_arrears()` for each loan, determining:

- **Days Past Due (DPD)**: calendar days from the earliest overdue instalment
- **Amount in Arrears**: sum of all overdue instalment amounts not yet paid
- **NPA Status**: true when DPD ≥ 90

### CBK Classification Matrix

| Classification | DPD | Provision Rate |
|---------------|-----|---------------|
| PERFORMING    | < 30 | 1% |
| WATCH         | 30–89 | 3% |
| SUBSTANDARD   | 90–179 | 20% |
| DOUBTFUL      | 180–359 | 50% |
| LOSS          | 360+ | 100% |

Run the nightly batch: `batch_calculate_arrears(tenant_id, as_of_date)` — this
is idempotent and safe to run multiple times per day.

---

## 6. Restructuring

Four restructure types are supported:

| Type | What it does |
|------|-------------|
| EXTEND_TENOR | Adds months to the loan tenor, regenerates schedule |
| REDUCE_RATE | Lowers interest rate, regenerates schedule |
| CAPITALISE_ARREARS | Rolls arrears + penalties into new principal |
| CONVERT_TO_TERM | Changes amortisation method (e.g., revolving → term) |

All restructures require `approved_by` and post a GL journal entry to the
restructure suspense account (3900).

---

## 7. Moratorium (Payment Holiday)

A moratorium suspends required payments for a defined period:

- **FULL**: No payments at all (interest may or may not accrue)
- **PRINCIPAL_ONLY**: Only interest is paid during the holiday

The schedule is regenerated from the day after moratorium ends.
The tenor is automatically extended by the moratorium duration.

---

## 8. Write-Off and Recovery

### Write-Off

When a loan is classified LOSS (360+ DPD) and approved by management:

```
DR Provision for Loan Losses (5100)   XXX
  CR Loans Receivable (1200)                XXX
```

The loan remains on the books (`WRITTEN_OFF` status) for recovery tracking.

### Recovery

Post write-off cash receipts:

```
DR Cash (1000)                        XXX
  CR Recovery Income (4200)                 XXX
```

If total recovery ≥ write-off amount, status transitions to `RECOVERED`.

---

## 9. Early Settlement

`get_early_settlement_amount()` returns:

| Field | Meaning |
|-------|---------|
| `outstanding` | Current balance + penalties + fees |
| `future_interest` | Total interest in remaining schedule |
| `rebate` | 50% rebate on future interest (configurable) |
| `settlement_amount` | outstanding − rebate |

---

## 10. Portfolio Quality Report

`get_portfolio_quality()` provides:

| Metric | Description |
|--------|-------------|
| `npl_ratio` | Non-performing loans / total portfolio |
| `par_30_ratio` | Portfolio at risk > 30 DPD |
| `par_90_ratio` | Portfolio at risk > 90 DPD |
| `provision_coverage` | Total posted provisions / NPL amount |
| `by_classification` | Breakdown by CBK classification |

---

## 11. Collections Workflow

1. **Reminder notice** — `send_demand_notice(..., REMINDER)` — automated SMS/email
2. **Formal demand** — `send_demand_notice(..., FORMAL_DEMAND)` — after 30 DPD
3. **Legal notice** — `send_demand_notice(..., LEGAL)` — after 90 DPD
4. **Collections referral** — `refer_to_collections()` — assigned to recovery team

---

## 12. New Features (v1.1.0)

### 12.1 Prepayment with Strategy

`prepay_with_options(tenant_id, loan_id, amount, prepay_date, payment_ref, strategy)`

The `strategy` parameter determines how the remaining schedule is rebuilt:

- `reduce_tenor` (default): Advance future principal amortisation, shorten the loan.
- `reduce_instalment`: Maintain original tenor but recalculate a lower constant PMT.
- `advance_next`: Same as reduce_tenor for schedule purposes; semantically applies to next installments.

The waterfall (penalties → fees → interest → principal) always runs first. Only excess principal triggers schedule regeneration.

### 12.2 IFRS 9 Daily Interest Accrual

`accrue_daily_interest(tenant_id, as_of_date)` is the end-of-day batch. It:

1. Loads all active loans.
2. Computes `outstanding_balance × annual_rate / 365 × days_since_last_accrual`.
3. Posts DR 1210 Accrued Interest Receivable / CR 4100 Interest Income.
4. Updates `last_accrual_date` on the loan record.

The job is idempotent: loans already accrued up to `as_of_date` are skipped.

### 12.3 Effective Interest Rate (EIR)

`calculate_eir(tenant_id, loan_id, origination_fees, transaction_costs)` solves for the monthly rate `r_m` satisfying:

```
net_proceeds = sum(CF_t / (1 + r_m)^t)
```

Using Newton-Raphson (200 iterations, tol=1e-8). Returns `eir_annual = (1 + r_m)^12 - 1`. Stores the result as an event for audit.

Use EIR as the discount rate when computing lifetime ECL (Stage 2/3).

### 12.4 IFRS 9 ECL Provisions

`compute_ecl_provision(tenant_id, loan_id, pd, lgd, ead, stage)`:

- **Stage 1**: `ECL = PD_12m × LGD × EAD`
- **Stage 2/3**: Lifetime ECL, summed over remaining installments with monthly PD and EIR discounting.

Typical staging rules:
- Stage 1: DPD < 30, no significant credit deterioration
- Stage 2: DPD 30–89, significant credit deterioration
- Stage 3: DPD ≥ 90, credit impaired (same as NPA)

### 12.5 Loan Top-Up

`topup_loan(tenant_id, loan_id, additional_amount, topup_date, approved_by, approved_limit)`:

Adds additional principal to an existing facility. Validates against `approved_limit` headroom. Regenerates the schedule from the topup date. Posts a `topup_disbursement` GL entry (DR 1200 / CR 2100). Emits a `topup` event for audit.

### 12.6 Collateral Management

**Register**: `register_collateral(tenant_id, loan_id, collateral_type, market_value, fsv, haircut_rate, valuation_date)`

Net Collateral Value = FSV × (1 − haircut_rate)

**Coverage**: `get_collateral_coverage(tenant_id, loan_id)` returns:
- `total_net_collateral_value`
- `coverage_ratio` = NCV / outstanding_balance
- `net_exposure` = max(0, outstanding − NCV)

Integrate with `calculate_required_provision` to apply CBK net-of-collateral provisioning.

### 12.7 Collections Escalation Ladder

`run_collections_escalation(tenant_id, as_of_date, policy)` runs a nightly idempotent escalation:

| DPD Threshold | Default Action |
|--------------|----------------|
| 5 | REMINDER notice |
| 15 | FORMAL_DEMAND notice |
| 30 | LEGAL notice |
| 60 | Refer to collections team |
| 90 | NPA (no notice; loan already reclassified) |

Override `policy` with a `{dpd_threshold: action_str}` dict for product-specific escalation.

### 12.8 Fee Schedule

`apply_fee(tenant_id, loan_id, fee_type, amount, due_date, defer_to_eir)`:

| Fee Type | IFRS 9 Treatment |
|----------|-----------------|
| ORIGINATION | Integral — defer, amortise via EIR |
| PROCESSING | Non-integral — expense immediately |
| ANNUAL_FACILITY | Expense at due date |
| EXIT / PREPAYMENT | Expense when triggered |
| INSURANCE | Expense immediately |

When `defer_to_eir=True`: DR Deferred Fee Asset (1250) / CR Fee Income (4300)
When `defer_to_eir=False`: DR Customer Account (2100) / CR Fee Income (4300)

### 12.9 CBK Regulatory Reporting

`generate_cbk_loan_register(tenant_id, reporting_date)` produces the CBK-LR1 loan register:
- All active loans with classification, DPD, required provision, posted provision, shortfall
- Summary totals (total portfolio, total required provisions)
- Data quality checks (missing customer_id, missing disbursement_date)

### 12.10 FX Loan Revaluation

`revalue_fx_loan(tenant_id, loan_id, spot_rate, revaluation_date, base_currency)`:

- No-op if loan is already in base currency.
- Computes delta = new_kes − prior_kes.
- Posts FX gain (DR 1200 / CR 3100) or FX loss (DR 3100 / CR 1200).
- Stores `last_revaluation_rate`, `last_revaluation_date`, and `kes_equivalent` on the loan.

Run monthly for every foreign-currency loan as part of month-end close.

---

## 14. Frequently Asked Questions

**Q: Can I use LMS without the GL capability?**  
A: Yes. LMS uses a `NullGLAdapter` by default — it prints journal lines to stdout
   rather than posting to the real GL. Inject a `GLAdapter` instance for production.

**Q: Is the batch arrears run safe to re-execute?**  
A: Yes — `batch_calculate_arrears()` and `accrue_daily_interest()` are fully idempotent.

**Q: What currency is supported?**  
A: Any ISO 4217 currency; defaults to KES (Kenya Shilling). FX loans use `revalue_fx_loan()`.

**Q: How are floating-point rounding errors avoided?**  
A: All arithmetic uses Python `Decimal` with `ROUND_HALF_UP`. All monetary fields
   are stored as `Decimal` in Pydantic models.

**Q: How do I integrate IFRS 9 ECL with the CBK prudential provisioning?**  
A: CBK requires the higher of the two. Call both `calculate_required_provision()` (CBK)
   and `compute_ecl_provision()` (IFRS 9), then post the larger amount via `post_provision_entry()`.

**Q: What is the difference between `reduce_tenor` and `reduce_instalment` prepayment?**  
A: `reduce_tenor` keeps the original instalment size and pays off the loan earlier.
   `reduce_instalment` keeps the original maturity date but lowers the periodic payment.
   Most retail lending frameworks default to `reduce_tenor`; some products (mortgages) prefer `reduce_instalment`.

---

© 2025 Datacraft | www.datacraft.co.ke | nyimbi@gmail.com
