# APG Point of Sale — Developer Guide
© 2025 Datacraft | www.datacraft.co.ke

## Architecture

```
capabilities/retail/pos/
├── models.py          Pydantic v2 models — all entities, enums, validators
├── service.py         PointOfSaleService — async business logic, 60+ methods
├── api.py             Flask Blueprint — REST API, url_prefix=/retail-pos/api/v1
├── views.py           Flask Blueprint — UI views, url_prefix=/retail-pos
├── app.py             Standalone Flask application factory
├── domain/
│   ├── rules.py       Pure business rules — assert_* and calculate_*
│   ├── calculations.py Pure financial calculations — no I/O
│   ├── events.py      Domain event dataclasses
│   └── adapters.py    Protocol definitions + null adapters for standalone mode
├── database/
│   └── schema.sql     PostgreSQL schema — normalized tables, partitioning hints
└── tests/
    └── test_service.py 67 tests — real objects, no mocks
```

## Coding Standards

- Python 3.12+, async throughout
- Tabs for indentation (never spaces)
- Modern typing: `str | None`, `list[str]`, `dict[str, Any]`
- Pydantic v2: `ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)`
- IDs: `str = Field(default_factory=uuid7str)` via `from uuid6 import uuid7`
- Service methods: all `async def`, return `dict[str, Any]`
- Logging: `_log_op()`, `_log_warn()`, `_log_txn()` helpers — never `print()`
- Domain events: `self._emit_event()` after every state change (wired to adapters)

## Service Initialisation

```python
from capabilities.retail.pos.service import PointOfSaleService

svc = PointOfSaleService()  # in-memory stores, null adapters

# Seed inventory for testing
svc._inventory.set_price("tenant-001", "SKU-MILK", 120.0)
svc._inventory.set_stock("store-nbi-01", "SKU-MILK", 100)
```

In production, swap `_InventoryStore` for a SQLAlchemy-backed adapter by injecting it at `__init__` time.

## Full Sale Flow

```python
import asyncio
from capabilities.retail.pos.service import PointOfSaleService
from capabilities.retail.pos.models import PosTerminalCreate

svc = PointOfSaleService()
T = "my-tenant"
STORE = "store-01"
CASHIER = "alice"

async def demo():
    # 1. Register terminal
    terminal = await svc.register_terminal(PosTerminalCreate(
        tenant_id=T, store_id=STORE, terminal_code="T001",
        terminal_type="fixed_counter", created_by="admin",
    ))

    # 2. Open session
    session = await svc.open_session(
        terminal_id=terminal["id"],
        cashier_id=CASHIER,
        opening_float=1000.0,
        tenant_id=T,
        store_id=STORE,
        created_by=CASHIER,
    )

    # 3. Seed prices
    svc._inventory.set_price(T, "MILK-1L", 120.0)

    # 4. Begin transaction
    txn = await svc.begin_transaction(
        session_id=session["id"],
        customer_id="customer-001",
        tenant_id=T,
        cashier_id=CASHIER,
        created_by=CASHIER,
    )

    # 5. Add items
    await svc.add_item(
        transaction_id=txn["id"],
        sku="MILK-1L",
        quantity=2,
        tenant_id=T,
        description="Whole Milk 1L",
        created_by=CASHIER,
    )

    # 6. Apply discount
    await svc.apply_discount(
        transaction_id=txn["id"],
        discount_type="percentage",
        value=10.0,
        approved_by="supervisor-bob",
        tenant_id=T,
    )

    # 7. Complete with split payment
    completed = await svc.complete_transaction(
        transaction_id=txn["id"],
        payments=[
            {"method": "cash", "amount": 100.0},
            {"method": "mpesa", "amount": 116.0, "reference": "QXZ123"},
        ],
        tenant_id=T,
        created_by=CASHIER,
    )
    print(f"Sale complete: {completed['transaction_number']} total={completed['grand_total']}")

    # 8. Generate receipt
    receipt = await svc.receipt_generation(
        transaction_id=completed["id"],
        fmt="thermal",
        tenant_id=T,
        created_by=CASHIER,
    )
    print(receipt["rendered_content"])

asyncio.run(demo())
```

## Adding a Custom Payment Adapter

```python
from capabilities.retail.pos.domain.adapters import AuthAdapter

class MyMpesaAdapter:
    async def initiate_stk_push(self, phone: str, amount: float, ref: str) -> dict:
        # Call Safaricom Daraja API
        ...
        return {"success": True, "reference": "QXZ456"}

# Wire in at service creation:
svc = PointOfSaleService()
svc._mpesa_adapter = MyMpesaAdapter()
```

## Domain Rules

All business rules are in `domain/rules.py` as standalone functions:

```python
from capabilities.retail.pos.domain.rules import (
    assert_session_open,
    assert_no_cross_tenant_access,
    assert_sufficient_payment,
    RuleViolation,
)

try:
    assert_session_open("closed")
except RuleViolation as e:
    print(e.rule_name)        # "session_not_open"
    print(e.required_action)  # "open_a_session"
```

Rules raise `RuleViolation`, not `AssertionError` — catch it explicitly in API handlers.

## Financial Calculations

All calculations are pure functions in `domain/calculations.py`:

```python
from capabilities.retail.pos.domain.calculations import (
    item_subtotal,
    item_tax,
    vat_inclusive_breakdown,
    expected_cash_in_till,
    suggest_denominations,
    top_selling_skus,
)

breakdown = vat_inclusive_breakdown(116.0, 0.16)
# {"net": 100.0, "vat": 16.0, "gross": 116.0}

change = suggest_denominations(750.0)
# {"500": 1, "200": 1, "50": 1}
```

## Multi-Tenancy

Every service method accepts `tenant_id` as a keyword argument. All store lookups filter by `(tenant_id, record_id)`. Cross-tenant access raises `RuleViolation("cross_tenant_access_denied")`.

In the API layer, `_tid()` extracts tenant from `g.tenant_id` (set by APG auth middleware) or the `X-Tenant-ID` request header (for standalone use).

## Testing

```bash
# Run all tests
python -m pytest capabilities/retail/pos/tests/ -v

# Run with coverage
python -m pytest capabilities/retail/pos/tests/ --cov=capabilities/retail/pos --cov-report=term-missing

# Type checking
python -m pyright capabilities/retail/pos/
```

Tests use:
- Real `PointOfSaleService` instances (no mocks)
- `asyncio.get_event_loop().run_until_complete()` (no `@pytest.mark.asyncio`)
- Class-based grouping by domain area
- Fixtures for terminal, session, and pre-built transaction

## Running Standalone

```bash
# Install
pip install -e capabilities/retail/pos/

# Run dev server
python -m capabilities.retail.pos --port 8080 --debug

# Or via entry point
apg-retail-pos --port 8080
```

Health check: `GET /health` → `{"status": "ok", "capability": "retail_pos"}`
Contract: `GET /contract` → full capability contract JSON

## Database

The `database/schema.sql` file creates the full normalized PostgreSQL schema under the `pos` schema:

```bash
psql $DATABASE_URL -f capabilities/retail/pos/database/schema.sql
```

Tables are partitioned by `tenant_id` (LIST partitioning) on high-volume entities (`pos_transactions`, `pos_payments`). Add explicit partitions for large tenants:

```sql
CREATE TABLE pos.pos_transactions_tenant_bigcorp
    PARTITION OF pos.pos_transactions
    FOR VALUES IN ('bigcorp');
```

The in-memory service stores are drop-in replacements for the DB layer — swap by implementing SQLAlchemy repositories that satisfy the same interface.

## APG Platform Integration

When running inside APG:
1. Register blueprint in `app.py` or APG composition engine
2. APG auth middleware sets `g.tenant_id` and `g.permissions`
3. The `has_access()` decorator in `views.py` enforces RBAC via `g.permissions`
4. Emit domain events via `domain/events.py` to APG's event bus

```python
# Register with APG composition engine
from capabilities.retail.pos.app import create_app
from capabilities.composition import register_capability

app = create_app()
register_capability("retail_pos", app, version="1.0.0")
```
