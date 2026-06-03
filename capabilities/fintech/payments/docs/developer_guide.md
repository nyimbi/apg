# APG Digital Payments — Developer Guide

**Version**: 2.0.0 | **Python**: 3.12+ | **Store**: In-memory (default) / PostgreSQL

---

## Architecture

```
fintech/payments/
├── models.py           # Pydantic v2 models — all entities
├── service.py          # DigitalPaymentsService — 60+ async methods
├── api.py              # Process-local API helpers (APG composition)
├── blueprint.py        # Flask Blueprint — REST API + UI routes
├── views.py            # Framework-neutral view models
├── capability_contract.py  # APG composition engine contract
├── domain/
│   ├── rules.py        # Pure business rule functions
│   ├── calculations.py # Pure financial calculations
│   ├── adapters.py     # Auth/Audit/Notify/Workflow protocols
│   └── events.py       # DomainEvent dataclass
├── database/
│   ├── schema.sql      # Full PostgreSQL schema (23 tables)
│   └── store.py        # InMemoryStore / PostgreSQLStore
└── tests/
    ├── test_models.py
    ├── test_rules.py
    ├── test_calculations.py
    ├── test_service.py
    └── test_blueprint.py
```

### Design Principles

1. **Tenant isolation first** — every method scopes to `tenant_id`; cross-tenant access raises `RuleViolation`
2. **Idempotency by default** — `idempotency_key` deduplicates at the service layer
3. **Pure domain** — `rules.py` and `calculations.py` have zero I/O; fully testable without infrastructure
4. **Pluggable adapters** — Auth, Audit, Notify, Workflow are Protocol interfaces; Null* variants auto-selected in standalone mode
5. **Store-agnostic** — switch from in-memory to PostgreSQL by passing `db_url`; same service code

---

## Service instantiation

```python
from capabilities.fintech.payments.service import DigitalPaymentsService

# Standalone (in-memory, no external deps)
svc = DigitalPaymentsService(tenant_id="acme", actor_id="user-42")

# Production (PostgreSQL)
svc = DigitalPaymentsService(
    tenant_id="acme",
    actor_id="user-42",
    db_url="postgresql+asyncpg://user:pass@host/db",
)

# With real APG adapters
from apg_common_auth import AuthService
svc = DigitalPaymentsService(
    tenant_id="acme",
    actor_id="user-42",
    auth=AuthService.from_env(),
)
```

---

## Key service methods

```python
# Mobile money
await svc.mpesa_stk_push(phone, amount, reference)
await svc.mpesa_b2c(phone, amount, occasion, remarks)
await svc.mpesa_b2b(business_short_code, amount, account_reference)
await svc.mtn_momo_request_to_pay(phone, amount, currency, external_id)
await svc.airtel_money_push(phone, amount, currency, reference)

# Card (PCI-DSS — token only)
await svc.card_authorise(card_token, amount, currency, merchant_id)
await svc.capture_payment(transaction_id, capture_amount)

# SWIFT / bank
await svc.swift_transfer(sender_bic, receiver_bic, iban, amount, currency)
await svc.bank_eft_transfer(from_account, to_account, bank_code, amount, ...)

# Batch
await svc.create_bulk_payment_batch(payment_date, method, currency, recipients, amounts, references)
await svc.validate_bulk_batch(batch_id)
await svc.process_bulk_batch(batch_id)

# FX
await svc.fx_convert(from_currency, to_currency, amount)
await svc.get_exchange_rate(from_currency, to_currency)

# Refunds / reversals
await svc.initiate_refund(transaction_id, amount, reason)
await svc.process_reversal(transaction_id, reason)

# Disputes
await svc.raise_dispute(transaction_id, reason, evidence_description)
await svc.investigate_dispute(dispute_id, investigation_notes)
await svc.resolve_chargeback(dispute_id, decision, chargeback_amount, decision_reason)

# Settlement
await svc.run_daily_settlement(settlement_date, bank_account)
await svc.reconcile_settlement(settlement_id, actual_amounts)

# Reporting
await svc.transaction_volume_report(period_from, period_to)
await svc.revenue_by_channel(period_from, period_to)
await svc.regulatory_transaction_report(period_from, period_to, regulator)
```

---

## Business rules (domain/rules.py)

All rules are callable pure functions returning `None` or raising `RuleViolation`:

```python
from capabilities.fintech.payments.domain.rules import (
    assert_mpesa_amount,
    assert_kyc_per_txn_limit,
    assert_no_duplicate,
    calculate_ctr_obligation,
    RuleViolation,
)

# Validate before calling service
try:
    assert_mpesa_amount(Decimal("5000"))
    assert_kyc_per_txn_limit(Decimal("5000"), "standard")
except RuleViolation as e:
    print(e.rule_name, e.reason, e.required_action)
```

---

## Financial calculations (domain/calculations.py)

```python
from capabilities.fintech.payments.domain.calculations import (
    mpesa_fee, fx_convert, settlement_net, velocity_score
)

fee = mpesa_fee(Decimal("5000"))
# FeeBreakdown(base_fee=57, excise_duty=11.40, total=68.40, currency='KES')

result = fx_convert(Decimal("100"), "USD", "KES", spread_bps=150)
# FXResult(from_amount=100, to_amount=12788.75, mid_rate=129.5, ...)

score = velocity_score(txn_count_24h=25, amount_sum_24h=Decimal("900000"),
                       unique_recipients_24h=15, failed_count_24h=2,
                       avg_amount=Decimal("5000"), current_amount=Decimal("100000"))
# {'score': 40, 'level': 'medium', 'flags': ['high_velocity', 'elevated_fan_out']}
```

---

## Flask Blueprint registration

```python
from flask import Flask
from capabilities.fintech.payments.blueprint import create_blueprint, create_ui_blueprint

app = Flask(__name__)
app.register_blueprint(create_blueprint(db_url=os.environ["DATABASE_URL"]))
app.register_blueprint(create_ui_blueprint())
```

All REST routes are under `/api/v1/payments/`. UI routes under `/payments/`.

---

## APG composition registration

```python
# capabilities/fintech/__init__.py
from capabilities.fintech.payments import get_capability_contract
from capabilities.composition import register_capability

register_capability("fintech_payments", get_capability_contract)
```

The contract exposes: `provides`, `requires`, `ui.routes`, `rule_engine.rules`, `streaming`, `configuration.agents`.

---

## Domain events

Every state change emits a `DomainEvent` to the capability event stream:

```python
from capabilities.fintech.payments.domain.events import DomainEvent

# Events emitted automatically by service:
# payment.initiated, payment.completed, payment.failed
# payment.refunded, payment.reversed
# dispute.opened, dispute.resolved
# settlement.complete
# fx.converted
```

Subscribe via the APG composition event bus or Bytewax stream processor.

---

## Testing

```bash
# All tests
python -m pytest capabilities/fintech/payments/tests/ -v

# Specific suite
python -m pytest capabilities/fintech/payments/tests/test_service.py -v

# Type check
uv run pyright capabilities/fintech/payments/

# Existing contract tests
python -m pytest capabilities/fintech/payments/tests/test_package_contract.py -v
```

No `@pytest.mark.asyncio` decorators — use `asyncio.run()` directly in tests.

---

## Extending the capability

### Add a new payment method

1. Add enum value to `PaymentMethod` in `models.py`
2. Add service method `async def new_method_pay(...)` in `service.py`
3. Add rule assertions in `domain/rules.py`
4. Add fee calculation in `domain/calculations.py`
5. Add Blueprint route in `blueprint.py`
6. Add tests in `tests/test_service.py` and `tests/test_blueprint.py`

### Add a new report

1. Add `async def my_report(period_from, period_to)` in `service.py`
2. Register as `GET /reports/my-report` in `blueprint.py`
3. Add APG API function in `api.py`

---

## Multi-tenancy

Every public service method signature includes `tenant_id`. The store filters all queries by tenant. Cross-tenant access raises `RuleViolation("cross_tenant_access_denied")`. Never pass user-controlled `tenant_id` without validating against the authenticated session.

---

## Performance notes

- PostgreSQL schema uses monthly range partitioning on `payment_transactions.created_at`
- Idempotency keys table enables O(1) dedup lookups
- JSONB GIN index on `apg_records.data` covers attribute filters in standalone mode
- For high-volume, add `customer_limits_usage` upserts for rolling daily/monthly limit checks instead of full table scans
