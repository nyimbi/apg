# Bank Account Management — Developer Guide

© 2025 Datacraft | `fin.acct` v1.0.0

## Architecture

```
capabilities/fin/acct/
├── service.py           # BankAccountService — all business logic
├── models.py            # Pydantic v2 models (enums, BankAccount, AccountTransaction, ...)
├── views.py             # Request/response models with AfterValidator guards
├── api.py               # Flask Blueprint (url_prefix=/api/fin/acct)
├── capability_contract.py  # Event stream, supported values, default config
├── domain/
│   └── adapters.py      # Abstract repository/event publisher protocols + in-memory impls
└── tests/
    ├── test_models.py
    └── test_service.py
```

## Service Design

`BankAccountService` stores all state in plain dicts (in-memory). Replace store attributes with repository objects implementing the protocols in `domain/adapters.py` for production.

```python
svc = BankAccountService(tenant_id="t1", user_id="u1")
acct = await svc.open_account("t1", "cust-001", "CURR001", "KES", opening_deposit=Decimal("5000"))
txn  = await svc.credit_account("t1", acct.id, Decimal("1000"), "KES", "REF-1", "Salary")
bal  = await svc.get_balance("t1", acct.id)
```

## Guard Pattern

Every public method calls guards before any processing:

```python
guard_tenant_id(tenant_id)           # non-empty, max 128 chars
guard_positive_amount(float(amount)) # > 0, finite, <= 1e12
guard_non_empty_string(reference)    # non-empty, max 65535
```

Guards are from `capabilities.common.reliability`.

## Monetary Arithmetic

All amounts use `Decimal` with `ROUND_HALF_UP` to 2dp. The `_d()` helper coerces any value:

```python
def _d(v: Any) -> Decimal:
    return Decimal(str(v)).quantize(TWO, rounding=ROUND_HALF_UP)
```

Never use `float` for money.

## GL Integration

Every credit/debit calls `_post_gl()` which emits a `gl_journal_requested` event on `apg.fin.glr.lifecycle`. A `CircuitBreaker` protects the call — if the circuit opens, the event is queued for retry. The `journal_id` returned is stored on the transaction record.

Production: replace `_post_gl` with an async NATS publish or direct `fin.glr` service call.

## Event Emission

`_log_event(event_type, payload)` writes to `self._events` (in-memory) and logs via `logging`. Production: drain `self._events` to NATS on `apg.fin.acct.lifecycle`.

## Balance Invariant

```
available_balance = book_balance - locked_balance + overdraft_available
overdraft_available = max(0, overdraft_limit - overdraft_used)
```

This invariant is maintained on every credit, debit, lock, and overdraft change.

## Adding a Product

```python
svc.products["PREM001"] = AccountProduct(
    product_code="PREM001",
    product_name="Premium Current Account",
    account_type=AccountType.CURRENT,
    currency="USD",
    overdraft_allowed=True,
    max_overdraft=Decimal("100000"),
    monthly_fee=Decimal("25"),
)
```

## Production Adapter Swap

```python
from capabilities.fin.acct.domain.adapters import (
    AccountRepository, TransactionRepository, EventPublisher, GLAdapter,
)

class PostgresAccountRepository(AccountRepository):
    async def save(self, account: dict) -> None:
        await db.execute("INSERT INTO ba_accounts ...", account)
    # ... implement all abstract methods

svc = BankAccountService()
svc.accounts = PostgresAccountRepositoryBackedDict(repo)
```

## Running Tests

```bash
cd /path/to/apg
python -m pytest capabilities/fin/acct/tests/ -v --tb=short
```

## Type Checking

```bash
uv run pyright capabilities/fin/acct/
```

## Flask Blueprint Registration

```python
from capabilities.fin.acct.api import bp
app.register_blueprint(bp)
# Routes available at /api/fin/acct/...
```

## APG Composition Registration

The `__init__.py` exports `CAPABILITY_META` which is consumed by APG's composition engine. The capability is auto-registered in `capabilities/fin/__init__.py`.

## Event Streams

| Event | Stream |
|-------|--------|
| account_opened, closed, frozen, dormant | `apg.fin.acct.lifecycle` |
| credit_posted, debit_posted | `apg.fin.acct.lifecycle` |
| transfer_completed | `apg.fin.acct.lifecycle` |
| gl_journal_requested | `apg.fin.glr.lifecycle` |
