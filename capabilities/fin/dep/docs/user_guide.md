# Deposit Products Engine — User Guide

**Capability**: `fin.dep` | **Version**: 1.0.0 | **Platform**: APG

---

## Overview

The Deposit Products Engine manages the complete lifecycle of deposit products for any bank or financial institution running on APG. It handles everything from product configuration through daily interest accrual, fee assessment, maturity processing, and regulatory reporting.

---

## Product Types

| Type | Use case | Interest | Fees |
|---|---|---|---|
| **CURRENT** | Transactional accounts | None or low | Maintenance fee |
| **SAVINGS** | Savings accounts | Tiered daily accrual | Minimum balance enforcement |
| **TERM_DEPOSIT** | Fixed-tenor deposits | Simple or compound | Early break penalty |
| **CALL_DEPOSIT** | Overnight / flexible | Daily accrual | None typically |
| **NOTICE_DEPOSIT** | Notice-period withdrawals | Daily accrual | Break penalty if notice not given |

---

## Creating a Product

A product defines the rules engine for all accounts of that type.

**Required fields**:
- `code` — unique per tenant (e.g. `SAV001`)
- `name` — human label
- `product_type` — see table above
- `currency` — ISO 4217 (e.g. `KES`, `USD`)
- `interest_config` — rate, method, tiers, WHT
- `fee_config` — maintenance fee, minimum balance, below-minimum fee
- `terms` — tenor limits, break penalty, tax exemption

**GL account mappings** (optional but recommended):
- `gl_interest_income_account` — P&L interest income account
- `gl_interest_payable_account` — balance sheet accrual account
- `gl_wht_payable_account` — WHT liability account

---

## Interest Calculation Methods

### SIMPLE
```
interest = principal × (rate/100) × (days/365)
```
Best for: term deposits with fixed tenor.

### DAILY_ACCRUAL
```
daily_interest = principal × (rate/100) / 365
total = daily_interest × days
```
Best for: savings and call deposits where balance may vary.

### COMPOUND
```
A = P × (1 + r/n)^(n×t)
interest = A - P
```
Where `n` = compounding periods per year (1=annually, 12=monthly, 365=daily).
Best for: high-yield savings products.

---

## Tiered Interest Rates

Tiers allow the applicable rate to increase with balance. The highest tier whose `min_balance` the account meets wins.

Example:
```
Tier 1: min_balance=0,      rate=3.0%   → balances < 10,000
Tier 2: min_balance=10000,  rate=4.5%   → balances 10,000–99,999
Tier 3: min_balance=100000, rate=6.0%   → balances ≥ 100,000
```

An account with balance 150,000 earns at 6.0%.

---

## Withholding Tax

Set `withholding_rate` in `interest_config` (e.g. `15` for 15%). The engine:
1. Computes gross interest
2. Deducts WHT: `wht = gross × (wht_rate / 100)`
3. Credits `net = gross - wht` to the account
4. Posts WHT liability to `gl_wht_payable_account`

Set `tax_exempt=True` in product terms to bypass WHT (e.g. government bonds).

---

## Nightly Batch Accrual

Run once per night per tenant:

```python
result = svc.batch_accrue_interest("bank1", date.today())
print(result.accounts_processed, result.total_accrued)
```

**Idempotent**: calling again with the same `(tenant_id, accrual_date)` returns the cached result without double-posting. Safe to retry on failure.

Accruals are stored as unposted entries. They become posted when `apply_interest()` is called.

---

## Term Deposit Maturity

At maturity, choose one of three instructions:

| Instruction | Effect |
|---|---|
| `ROLLOVER` | Interest applied, principal + interest reinvested |
| `PAYOUT` | Interest applied, full balance transferred out, account zeroed |
| `PARTIAL` | Interest applied, specified amount paid out, remainder stays |

```python
svc.process_term_deposit_maturity("bank1", "ACC-001", MaturityInstruction.ROLLOVER)
```

---

## Early Break Penalty

For TERM_DEPOSIT and NOTICE_DEPOSIT products, configure `break_penalty_rate` as a percentage of gross interest forfeited:

```python
terms=ProductTerms(break_penalty_rate=Decimal("50"))  # 50% of interest forfeited
```

Check the cost before breaking:
```python
penalty = svc.calculate_break_penalty("bank1", "ACC-001", date.today())
```

---

## Maturity Simulation (What-If)

No state changes — pure projection:

```python
result = svc.simulate_maturity("bank1", "TD001", principal=Decimal("200000"), tenor_days=90)
print(result.maturity_amount, result.effective_rate)
```

---

## Fee Assessment

### Maintenance fee
Charged monthly or quarterly regardless of balance (if configured).

### Below-minimum fee
Charged instead of (not in addition to) the maintenance fee when balance falls below `minimum_balance`.

```python
check = svc.check_minimum_balance("bank1", "ACC-001")
if not check.meets_minimum:
    svc.apply_maintenance_fee("bank1", "ACC-001", date.today())
```

---

## Rate Changes

Rate changes take effect immediately but are fully audited:

```python
svc.update_product_rate(
    "bank1", "SAV001",
    new_rate=Decimal("7.0"),
    effective_date=date(2025, 7, 1),
    reason="CBK rate adjustment",
)
schedule = svc.get_rate_schedule("bank1", "SAV001")  # full history
```

---

## Withholding Tax Reporting

Generate a period report for tax filing:

```python
# Monthly
report = svc.get_withholding_tax_report("bank1", "2025-03")

# Quarterly
report = svc.get_withholding_tax_report("bank1", "2025-Q1")
```

Each entry contains `account_id`, `gross_amount`, `wht_amount`, and `posted_at`.

---

## Common Workflows

### New savings account
1. `create_product()` — define the product once
2. `register_account()` — attach an account to the product
3. `batch_accrue_interest()` — run nightly
4. `apply_interest()` — post interest monthly
5. `check_minimum_balance()` + `apply_maintenance_fee()` — monthly fee cycle

### Term deposit lifecycle
1. `create_product()` with `min_tenor_days`, `break_penalty_rate`
2. `register_account()` with `opening_date` and `maturity_date`
3. `batch_accrue_interest()` — daily
4. At maturity: `process_term_deposit_maturity()`
5. If early break: `calculate_break_penalty()` → inform customer → manual posting

---

---

## v1.1 New Features

### Product Cloning

Quickly create product variants without re-entering all configuration:

```python
await svc.clone_product(
    tenant_id="bank1",
    source_code="SAV001",
    new_code="SAV-PREMIUM",
    new_name="Premium Savings",
    overrides={
        "interest_config": InterestConfig(rate=Decimal("8.5"), ...),
    },
    cloned_by="product_manager",
)
```

The clone inherits all fields from the source product. Only fields listed in `overrides` are replaced. Rate history is initialised with a `cloned_from:<source_code>` entry.

---

### Multi-Product Comparison

Compare returns across multiple products in a single call:

```python
rankings = await svc.compare_products(
    "bank1",
    principal=Decimal("1000000"),
    tenor_days=90,
    product_codes=["TD-6M", "TD-3M", "SAV-PREMIUM"],
)
# rankings[0] is the product with highest net_interest
for r in rankings:
    print(r.product_code, r.net_interest, r.effective_rate)
```

Products that cannot be simulated (inactive, wrong type) are silently excluded.

---

### Effective Annual Yield (EAY)

Mandatory CBK/CMA disclosure — the true annual return after compounding and tax:

```python
eay = await svc.get_effective_annual_yield(
    "bank1", "TD-6M", principal=Decimal("500000"),
    tax_rate_override=Decimal("15"),  # optional override
)
print(eay["disclosure_text"])
# → "Gross EAY 9.3807% | WHT 15% | Net EAY 7.9736% on KES 500,000.00 principal"
```

Formula used:
- Compound: `gross_eay = (1 + r/n)^n - 1`
- Simple / daily: `gross_eay = r` (nominal rate already annual)
- `net_eay = gross_eay × (1 - wht_rate/100)`

---

### Dormancy Management

CBK Prudential Guideline CBK/PG/01 requires classification of inactive accounts:

```python
# Run at month-end; accounts idle >= 365 days become dormant
report = await svc.classify_dormant_accounts("bank1", date.today(), inactivity_days=365)
print(report["newly_dormant"], report["fees_applied"])

# Reactivate when customer transacts
await svc.reactivate_account("bank1", "ACC-001", reactivated_by="teller_001")
```

Dormancy classification:
1. Finds accounts with no interest postings for `inactivity_days`
2. Marks them `dormant=True` with `dormant_since` timestamp
3. Applies the configured maintenance fee

---

### Batch Maturity Sweep

Process all term deposits maturing on or before a given date in a single EOD call:

```python
report = await svc.batch_process_maturities(
    "bank1",
    maturity_date=date.today(),
    default_instruction=MaturityInstruction.ROLLOVER,
)
print(report["processed_count"], report["total_interest"], report["errors"])
```

Each account uses its per-account `maturity_instruction` if set, otherwise falls back to `default_instruction`. Errors are collected per-account — a single failure does not halt the batch.

---

### Accrual Reversal

Correct GL divergence caused by rate errors or backdated transactions:

```python
result = await svc.reverse_accrual(
    "bank1", "ACC-001",
    accrual_date=date(2026, 5, 31),
    reason="rate_correction_post_cbr_change",
    reversed_by="finance_ops",
)
print(result["net_reversed"], result["reversed_at"])
```

A negating `AccrualEntry` is stored under a `REV:` key. The original entry is marked as posted to prevent double-accrual. The reversal record links back to the original via `posting_ref`.

---

### Account Statement

Generate a period statement with running balance (CBK Banking Act s.24):

```python
stmt = await svc.generate_account_statement(
    "bank1", "ACC-001",
    from_date=date(2026, 1, 1),
    to_date=date(2026, 3, 31),
)
print(stmt["opening_balance"])
for item in stmt["line_items"]:
    print(item["date"], item["type"], item["amount"], item["running_balance"])
print(stmt["closing_balance"])
```

Statement includes:
- `INTEREST_CREDIT` entries from `apply_interest()`
- `FEE_DEBIT` entries from `apply_maintenance_fee()`
- Running balance after each line item
- Opening and closing balance

---

### Interest Disposition

Private banking clients may prefer interest paid to a linked current account rather than capitalised:

```python
# Set pay-out mode
await svc.set_interest_disposition(
    "bank1", "ACC-001",
    disposition="PAY_OUT",
    linked_payout_account="CHQ-001",
)

# Revert to capitalisation (default)
await svc.set_interest_disposition("bank1", "ACC-001", "CAPITALIZE")
```

`apply_interest()` respects this flag: when `PAY_OUT`, the linked account receives the net interest credit instead of the deposit account.

---

## APG Integration Notes

- Auth and permissions flow through `common.auth_rbac`
- All significant events are audited via `common.audit_compliance`
- GL postings route to `fin.glr` via the GL adapter
- The `NullGLAdapter` is used automatically in standalone/test mode
