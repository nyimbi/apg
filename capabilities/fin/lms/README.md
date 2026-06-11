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

## Testing

```bash
uv run pytest -vxs capabilities/fin/lms/tests/
```

---

## License

© 2025 Datacraft. All rights reserved.  
Contact: nyimbi@gmail.com | www.datacraft.co.ke
