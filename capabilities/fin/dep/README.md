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
