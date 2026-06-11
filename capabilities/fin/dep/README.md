# fin.dep — Deposit Products Engine

Banking product factory and interest calculation engine for deposit accounts.

© 2025 Datacraft · Author: Nyimbi Odero

---

## What it does

| Feature | Details |
|---|---|
| **Product types** | CURRENT, SAVINGS, TERM_DEPOSIT, CALL_DEPOSIT, NOTICE_DEPOSIT |
| **Interest methods** | SIMPLE, COMPOUND (daily/monthly/annual compounding), DAILY_ACCRUAL |
| **Tiered rates** | Unlimited tiers; highest qualifying tier wins |
| **Withholding tax** | Per-product WHT rate, deducted at posting |
| **Fee engine** | Monthly/quarterly maintenance fee, below-minimum penalty |
| **Term deposit** | Maturity processing: ROLLOVER / PAYOUT / PARTIAL |
| **Early break** | Configurable penalty as % of gross interest forfeited |
| **Batch accrual** | Nightly idempotent accrual for all accounts |
| **Simulation** | What-if projection without touching state |
| **Rate history** | Full audit trail of every rate change |
| **WHT reporting** | Monthly or quarterly withholding tax reports |

---

## Quick start (standalone)

```python
from decimal import Decimal
from datetime import date
from capabilities.fin.dep.service import DepositProductsService
from capabilities.fin.dep.models import (
    ProductType, InterestCalculationType, CompoundingFrequency,
    FeeFrequency, InterestConfig, FeeConfig, ProductTerms, MaturityInstruction,
)

svc = DepositProductsService()

# Create a savings product
svc.create_product(
    tenant_id="bank1",
    code="SAV001",
    name="Classic Savings",
    product_type=ProductType.SAVINGS,
    currency="KES",
    interest_config=InterestConfig(
        rate=Decimal("6.5"),
        calculation=InterestCalculationType.DAILY_ACCRUAL,
        compounding=CompoundingFrequency.MONTHLY,
        withholding_rate=Decimal("15"),
    ),
    fee_config=FeeConfig(
        maintenance_fee=Decimal("200"),
        fee_frequency=FeeFrequency.MONTHLY,
        minimum_balance=Decimal("1000"),
        below_minimum_fee=Decimal("50"),
    ),
    terms=ProductTerms(min_opening_amount=Decimal("500")),
)

# Register an account
svc.register_account("bank1", "ACC-001", "SAV001", Decimal("50000"), date(2025, 1, 1))

# Calculate interest for Q1
result = svc.calculate_interest(
    "bank1", "ACC-001",
    date(2025, 1, 1), date(2025, 3, 31),
    Decimal("50000"), "SAV001",
)
print(result.gross_interest, result.withholding_tax, result.net_interest)

# Nightly batch accrual (idempotent)
batch = svc.batch_accrue_interest("bank1", date(2025, 3, 15))
print(batch.accounts_processed, batch.total_accrued)
```

---

## New Features (v1.1)

| Feature | Method | Improvement |
|---|---|---|
| **Product Cloning** | `clone_product()` | Deep-copy any product with optional field overrides; full rate-history init |
| **Multi-Product Comparison** | `compare_products()` | Fan-out simulation across N products in one call, ranked by net interest |
| **Effective Annual Yield** | `get_effective_annual_yield()` | EAY with WHT, mandatory CBK/CMA disclosure text |
| **Dormancy Classification** | `classify_dormant_accounts()` | CBK/PG/01 dormancy sweep with fee assessment |
| **Dormancy Reactivation** | `reactivate_account()` | Reverse dormant classification with audit trail |
| **Batch Maturity Sweep** | `batch_process_maturities()` | EOD sweep for all term deposits maturing on/before a date |
| **Accrual Reversal** | `reverse_accrual()` | Negating GL entry for corrections; prevents ledger divergence |
| **Account Statement** | `generate_account_statement()` | Opening balance, line items, running balance, closing balance |
| **Interest Disposition** | `set_interest_disposition()` | CAPITALIZE or PAY_OUT to a linked account per depositor |

### Usage examples (v1.1)

```python
import asyncio
from decimal import Decimal
from datetime import date
from capabilities.fin.dep.service import DepositProductsService

svc = DepositProductsService()

# Clone a product with a higher rate
async def demo():
    await svc.clone_product(
        "bank1", "SAV001", "SAV-PREMIUM", "Premium Savings",
        overrides={"interest_config": interest_cfg_8pct},
    )

    # Compare 3 products for a 90-day, KES 500,000 deposit
    rankings = await svc.compare_products("bank1", Decimal("500000"), 90,
        ["TD001", "TD002", "TD003"])
    best = rankings[0]  # highest net interest first

    # Regulatory yield disclosure
    eay = await svc.get_effective_annual_yield("bank1", "TD001", Decimal("500000"))
    print(eay["disclosure_text"])

    # EOD dormancy sweep
    dormancy_report = await svc.classify_dormant_accounts("bank1", date.today(), 365)

    # EOD maturity sweep
    maturity_report = await svc.batch_process_maturities("bank1", date.today())

    # Reverse a bad accrual
    await svc.reverse_accrual("bank1", "ACC-001", date(2026, 5, 31),
        reason="rate_correction", reversed_by="ops_team")

    # Generate statement for Q1 2026
    stmt = await svc.generate_account_statement(
        "bank1", "ACC-001", date(2026, 1, 1), date(2026, 3, 31))
    print(stmt["opening_balance"], stmt["closing_balance"])

    # Set interest pay-out to a linked current account
    await svc.set_interest_disposition(
        "bank1", "ACC-001", "PAY_OUT", linked_payout_account="CHQ-001")

asyncio.run(demo())
```

---

## API reference summary

All API functions live in `capabilities.fin.dep.api` and accept plain dicts.

| Function | Description |
|---|---|
| `create_product(payload)` | Define a new deposit product |
| `get_product(tenant_id, code)` | Fetch product definition |
| `list_products(payload)` | List products, optionally filtered |
| `update_product(payload)` | Mutate name/configs |
| `deactivate_product(tenant_id, code)` | Soft-deactivate |
| `calculate_interest(payload)` | Compute interest for a period |
| `apply_interest(payload)` | Post interest credit + GL journal |
| `apply_maintenance_fee(payload)` | Post monthly/quarterly fee |
| `check_minimum_balance(tenant_id, account_id)` | Minimum balance check |
| `process_term_deposit_maturity(payload)` | Rollover / payout at maturity |
| `get_accrued_interest(tenant_id, account_id, as_of_date)` | Unposted accruals |
| `calculate_break_penalty(tenant_id, account_id, break_date)` | Early break cost |
| `get_interest_history(payload)` | Posted interest history |
| `get_rate_schedule(tenant_id, product_code)` | Full rate change history |
| `update_product_rate(payload)` | Change rate with effective date |
| `get_products_by_balance(payload)` | Products eligible for a given balance |
| `simulate_maturity(payload)` | What-if projection |
| `get_product_stats(tenant_id)` | Portfolio summary |
| `batch_accrue_interest(payload)` | Idempotent nightly accrual |
| `get_withholding_tax_report(tenant_id, period_id)` | WHT report by period |
| `health()` | Service health check |
| `clone_product(payload)` | Deep-copy product with optional overrides |
| `compare_products(payload)` | Multi-product simulation ranked by net interest |
| `get_effective_annual_yield(payload)` | EAY with WHT for regulatory disclosure |
| `classify_dormant_accounts(payload)` | CBK dormancy sweep with fee assessment |
| `reactivate_account(tenant_id, account_id)` | Reverse dormant classification |
| `batch_process_maturities(payload)` | EOD sweep for all maturing term deposits |
| `reverse_accrual(payload)` | GL-correcting accrual reversal |
| `generate_account_statement(payload)` | Full period statement with running balance |
| `set_interest_disposition(payload)` | CAPITALIZE or PAY_OUT interest per account |

---

## Tests

```bash
python -m pytest capabilities/fin/dep/tests/ -v
```

72 tests, all passing.

---

## APG Integration

- **Requires**: `fin.glr` (GL posting), `common.auth_rbac`, `common.audit_compliance`
- **Event streams**: `apg.fin.dep.interest`, `apg.fin.dep.maturity`, `apg.fin.dep.fee`
- **API prefix**: `/api/fin/dep`
- **DB table prefix**: `dep_`
- **Standalone**: works with null adapters — no APG platform required for dev/test
