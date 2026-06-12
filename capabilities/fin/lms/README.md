# Loan Management System (LMS)

**Capability ID**: `fin_lms`  
**Domain**: Financial (`fin`)  
**Version**: 1.0.0  
**Author**: Nyimbi Odero — Datacraft © 2025

---

## Overview

LMS is the post-origination loan lifecycle engine for the APG platform.
`fintech_lending` handles credit origination and approval; once a loan is
approved, LMS takes over from disbursement through to full closure.

```
fintech_lending  →  [APPROVED]  →  fin_lms
                                    ├── disburse
                                    ├── generate schedule
                                    ├── record repayments (waterfall)
                                    ├── calculate arrears / NPA
                                    ├── classify (CBK/Basel)
                                    ├── provision (CBK matrix)
                                    ├── restructure / moratorium
                                    ├── write-off / recovery
                                    └── close
```

---

## Supported Amortisation Methods

| Method | Description |
|--------|-------------|
| `REDUCING_BALANCE` | Equal principal instalments, diminishing interest |
| `FLAT_RATE` | Interest computed on original principal throughout |
| `FRENCH_ANNUITY` | Constant PMT payment (standard mortgage) |
| `BULLET` | Interest monthly, full principal at maturity |
| `INTEREST_ONLY` | Interest only (alias for BULLET) |

---

## Repayment Waterfall

Payments are applied in priority order:

1. **Penalties** (late fees / daily penalty accruals)
2. **Fees**
3. **Interest** (oldest instalment first)
4. **Principal** (oldest instalment first)

---

## CBK Loan Classification (DPD thresholds)

| Classification | Days Past Due | CBK Provision Rate |
|---------------|--------------|-------------------|
| PERFORMING    | 0–29         | 1%                |
| WATCH         | 30–89        | 3%                |
| SUBSTANDARD   | 90–179       | 20%               |
| DOUBTFUL      | 180–359      | 50%               |
| LOSS          | 360+         | 100%              |

---

## Quick Start

```python
import asyncio
from datetime import date
from decimal import Decimal
from capabilities.fin.lms import LoanManagementService, LoanStatus, AmortisationMethod
from capabilities.fin.lms.models import Loan

async def demo():
    svc = LoanManagementService()

    # 1. Create a loan record (normally supplied by fintech_lending)
    loan = Loan(
        tenant_id="bank1",
        customer_id="cust-001",
        product_code="TERM-12M",
        principal=Decimal("100000"),
        rate=Decimal("0.14"),       # 14% p.a.
        tenor_months=12,
        method=AmortisationMethod.REDUCING_BALANCE,
    )
    await svc._loans.save(loan.model_dump())

    # 2. Disburse
    result = await svc.disburse_loan(
        tenant_id="bank1",
        loan_id=loan.id,
        disbursement_date=date(2025, 1, 15),
        account_id="ACC-001",
        amount=Decimal("100000"),
        disbursement_ref="DRF-2025-001",
    )
    print(result["disbursed_amount"])   # 100000

    # 3. Record a repayment
    from capabilities.fin.lms.models import PaymentMethod
    rp = await svc.record_repayment(
        tenant_id="bank1",
        loan_id=loan.id,
        amount=Decimal("9500"),
        payment_date=date(2025, 2, 15),
        payment_ref="PAY-001",
        payment_method=PaymentMethod.MOBILE_MONEY,
    )
    print(rp["remaining_balance"])

asyncio.run(demo())
```

---

## API Endpoints (`/api/fin/lms`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/loans/{id}/disburse` | Disburse a loan |
| POST | `/schedule/generate` | Generate amortisation schedule |
| POST | `/loans/{id}/repayments` | Record repayment |
| POST | `/loans/{id}/arrears` | Calculate arrears position |
| POST | `/loans/{id}/penalty` | Apply penalty charge |
| POST | `/batch/arrears` | Nightly batch arrears run |
| POST | `/loans/{id}/restructure` | Restructure loan |
| POST | `/loans/{id}/moratorium` | Grant payment moratorium |
| POST | `/loans/{id}/reprice` | Change interest rate |
| POST | `/loans/{id}/write-off` | Write off loan |
| POST | `/loans/{id}/recovery` | Record post write-off recovery |
| GET  | `/loans/{id}` | Full loan details |
| GET  | `/loans/{id}/schedule` | Amortisation schedule |
| GET  | `/loans/{id}/statement` | Loan statement |
| GET  | `/loans` | List loans (filtered) |
| GET  | `/portfolio/quality` | Portfolio quality metrics |
| GET  | `/loans/{id}/classification` | CBK classification |
| GET  | `/loans/{id}/provision/required` | Required provision amount |
| POST | `/loans/{id}/provision` | Post provision entry |
| GET  | `/provision/report` | Full provision report |
| POST | `/loans/{id}/notice` | Send demand notice |
| POST | `/loans/{id}/collections/refer` | Refer to collections |
| POST | `/loans/{id}/close` | Close loan |
| GET  | `/loans/{id}/early-settlement` | Early settlement amount |
| GET  | `/health` | Health check |

---

## GL Account Codes Used

| Code | Account | DR/CR |
|------|---------|-------|
| 1000 | Cash | DR (recovery) |
| 1200 | Loans Receivable | DR (disbursement), CR (repayment/write-off) |
| 1290 | Allowance for Loan Losses | CR (provision) |
| 2100 | Customer Account | CR (disbursement), DR (repayment) |
| 3900 | Restructure Suspense | CR |
| 4200 | Recovery Income | CR |
| 5100 | Provision for Loan Losses | DR (write-off/provision) |
| 6100 | Provision Expense | DR |

---

## New Features (v1.1.0)

### Prepayment with Strategy (`prepay_with_options`)

Allows borrowers or RMs to control how surplus principal prepayment reshapes the loan:

| Strategy | Behaviour |
|----------|-----------|
| `reduce_tenor` | Advance future principal, shorten remaining term |
| `reduce_instalment` | Keep tenor, recalculate lower PMT (French annuity) |
| `advance_next` | Apply surplus to the next installment(s) in sequence |

```python
result = await svc.prepay_with_options(
    tenant_id="bank1",
    loan_id=loan.id,
    amount=Decimal("20000"),
    prepay_date=date(2025, 6, 1),
    payment_ref="PRE-001",
    strategy="reduce_instalment",
)
print(result["remaining_balance"])  # lower balance, same tenor
```

### Daily Interest Accrual — IFRS 9 (`accrue_daily_interest`)

Batch posts daily `balance × rate / 365` accruals for all active loans:
  DR Accrued Interest Receivable (1210) / CR Interest Income (4100)

```python
summary = await svc.accrue_daily_interest(tenant_id="bank1", as_of_date=date.today())
print(summary["total_accrued"])
```

### Effective Interest Rate / XIRR (`calculate_eir`)

Newton-Raphson solver over the full cashflow stream incorporating origination fees.
Required for IFRS 9 amortised cost measurement.

```python
eir = await svc.calculate_eir(
    tenant_id="bank1",
    loan_id=loan.id,
    origination_fees=Decimal("2000"),
    transaction_costs=Decimal("500"),
)
print(eir["eir_annual"])   # e.g. "0.1512"
```

### IFRS 9 ECL Stage Bucketing (`compute_ecl_provision`)

Computes 12-month (Stage 1) or lifetime (Stage 2/3) Expected Credit Loss:

```python
ecl = await svc.compute_ecl_provision(
    tenant_id="bank1",
    loan_id=loan.id,
    pd=Decimal("0.05"),   # 5% annual PD
    lgd=Decimal("0.40"),  # 40% LGD
    stage=1,
)
print(ecl["ecl"])
```

### Loan Top-Up / Additional Drawdown (`topup_loan`)

Add principal to an existing facility without re-origination:

```python
topup = await svc.topup_loan(
    tenant_id="bank1",
    loan_id=loan.id,
    additional_amount=Decimal("50000"),
    topup_date=date(2025, 7, 1),
    approved_by="rm_alice",
    approved_limit=Decimal("200000"),
)
```

### Collateral Registration and Coverage (`register_collateral`, `get_collateral_coverage`)

Track FSV-based collateral for CBK net-of-collateral provisioning:

```python
await svc.register_collateral(
    tenant_id="bank1",
    loan_id=loan.id,
    collateral_type="land_title",
    market_value=Decimal("150000"),
    fsv=Decimal("120000"),
    haircut_rate=Decimal("0.10"),
    valuation_date=date(2025, 1, 1),
)
coverage = await svc.get_collateral_coverage(tenant_id="bank1", loan_id=loan.id)
print(coverage["coverage_ratio"])   # e.g. "1.0800"
```

### Automated Collections Escalation (`run_collections_escalation`)

Nightly idempotent ladder: REMINDER → FORMAL_DEMAND → LEGAL → COLLECTIONS_REFERRAL:

```python
result = await svc.run_collections_escalation(
    tenant_id="bank1",
    as_of_date=date.today(),
)
print(result["escalated"])
```

### Fee Schedule Engine (`apply_fee`)

Structured fee application with IFRS 9 deferral support:

```python
await svc.apply_fee(
    tenant_id="bank1",
    loan_id=loan.id,
    fee_type="ORIGINATION",
    amount=Decimal("2000"),
    due_date=date(2025, 1, 15),
    defer_to_eir=True,   # deferred, amortised over tenor
)
```

### CBK Regulatory Reporting (`generate_cbk_loan_register`)

Generate CBK Form CBK-LR1 loan register with data quality checks:

```python
register = await svc.generate_cbk_loan_register(
    tenant_id="bank1",
    reporting_date=date(2025, 3, 31),
)
print(register["total_provisions_required"])
for row in register["rows"]:
    print(row["loan_id"], row["classification"], row["required_provision"])
```

### FX Loan Revaluation (`revalue_fx_loan`)

Translate foreign-currency loan balances to KES at spot rate:

```python
result = await svc.revalue_fx_loan(
    tenant_id="bank1",
    loan_id=usd_loan.id,
    spot_rate=Decimal("130.50"),  # USD/KES
    revaluation_date=date(2025, 3, 31),
)
print(result["fx_gain_loss"], result["gain_or_loss"])
```

---

## Updated API Endpoints (`/api/fin/lms`)

New endpoints added in v1.1.0:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/loans/{id}/prepay` | Prepayment with strategy control |
| POST | `/batch/accrue-interest` | IFRS 9 daily accrual batch |
| GET  | `/loans/{id}/eir` | Calculate EIR / XIRR |
| POST | `/loans/{id}/ecl` | Compute IFRS 9 ECL provision |
| POST | `/loans/{id}/topup` | Loan top-up / additional drawdown |
| POST | `/loans/{id}/collateral` | Register collateral |
| GET  | `/loans/{id}/collateral/coverage` | Collateral coverage metrics |
| POST | `/batch/collections-escalation` | Run collections escalation ladder |
| POST | `/loans/{id}/fees` | Apply structured fee |
| GET  | `/reports/cbk-loan-register` | CBK Form CBK-LR1 |
| POST | `/loans/{id}/fx-revalue` | FX revaluation for foreign currency loans |

---

## Testing

```bash
uv run pytest -vxs capabilities/fin/lms/tests/
```

---

## License

© 2025 Datacraft. All rights reserved.  
Contact: nyimbi@gmail.com | www.datacraft.co.ke

---

## World-Class Enhancements (v2.0)

- **I1.** LMS World-Class Improvements
- **I2.** Partial Prepayment with Configurable Waterfall Override
- **I3.** Interest Accrual Engine (Daily Accrual, Month-End Posting)
- **I4.** Effective Interest Rate (EIR) / XIRR Calculation
- **I5.** Expected Credit Loss (ECL) — IFRS 9 Stage Bucketing
- **I6.** Covenant Monitoring and Breach Alerts
- **I7.** Instalment-Level Partial Pay Tracking (Split Installment Clearing)
- **I8.** Loan Top-Up (Additional Drawdown on Existing Facility)
- **I9.** Collateral Tracking and Forced Sale Value
- **I10.** Collections Workflow Automation (Escalation Ladder)
- **I11.** Fee Schedule Engine (Disbursement, Processing, Annual, Exit Fees)
- **I12.** Bulk Portfolio Operations with Idempotency Keys
- **I13.** Loan Participations / Syndication Splits
- **I14.** Regulatory Reporting Pack (CBK, CRB, IFRS 9 Disclosures)
- **I15.** Multi-Currency Loan Support with FX Revaluation

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
