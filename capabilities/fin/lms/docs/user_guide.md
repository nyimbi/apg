# Loan Management System — User Guide

**Version**: 1.0.0 | **Platform**: APG `fin` domain  
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
- Calculates and posts required provisions per CBK matrix
- Handles restructuring, moratoriums, write-offs, and recoveries
- Produces portfolio quality reports (NPL, PAR>30, PAR>90, provision coverage)
- Sends demand notices and manages collections referrals
- Supports early settlement with interest rebate calculation

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

## 12. Frequently Asked Questions

**Q: Can I use LMS without the GL capability?**  
A: Yes. LMS uses a `NullGLAdapter` by default — it prints journal lines to stdout
   rather than posting to the real GL. Inject a `GLAdapter` instance for production.

**Q: Is the batch arrears run safe to re-execute?**  
A: Yes — `batch_calculate_arrears()` is fully idempotent.

**Q: What currency is supported?**  
A: Any ISO 4217 currency; defaults to KES (Kenya Shilling).

**Q: How are floating-point rounding errors avoided?**  
A: All arithmetic uses Python `Decimal` with `ROUND_HALF_UP`. All monetary fields
   are stored as `Decimal` in Pydantic models.

---

© 2025 Datacraft | www.datacraft.co.ke | nyimbi@gmail.com
