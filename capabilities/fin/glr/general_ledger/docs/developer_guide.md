# General Ledger — Developer Guide

© 2025 Datacraft. Author: Nyimbi Odero

---

## Architecture overview

The GLR module is a pure-Python, dependency-light double-entry accounting engine that ships as a first-class APG capability. It deliberately separates concerns into five layers:

```
api.py          ← Flask Blueprint REST interface (url_prefix /api/glr)
views.py        ← Screen models — pure dicts, no HTTP concerns
service.py      ← Business logic, in-memory stores, async methods
domain/
  rules.py      ← Pure rule functions, RuleViolation exception
  calculations.py ← Pure math — no I/O
  adapters.py   ← Null-safe adapters for auth, audit, notify, workflow
database/
  schema.sql    ← PostgreSQL DDL, full production schema
  store.py      ← Store abstraction (in-memory default, SQL optional)
models.py       ← Pydantic v2 request/response models
blueprint.py    ← APG composition engine registration
capability_contract.py ← Rule engine + streaming contract
```

### APG composition integration

`blueprint.py` exposes `init_subcapability(app)` and `register_with_composition_engine(registry)`. The APG orchestrator calls these during capability graph construction.

The capability contract in `capability_contract.py` defines:
- Event stream: `apg.fin.glr.lifecycle` via Bytewax
- Provided contracts: `chart_of_accounts_lifecycle`, `journal_posting_workflow`, `glr_agents`
- Dependencies: `fin.apy`, `fin.arc`, `fin.cbm`, `fin.fam`, `fin.bfc`, `fin.txm`, `fin.fco`

---

## Coding standards

All code follows the project CLAUDE.md conventions:

- **Python 3.12+**, async throughout
- **Tabs** for indentation (never spaces)
- **Modern typing**: `str | None`, `list[str]`, `dict[str, Any]`
- **Pydantic v2**: `ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)`
- **IDs**: `str = Field(default_factory=uuid7str)` using `uuid6.uuid7`
- **Logging**: `_log_*` prefixed private methods
- **Runtime assertions** at function entry and exit

---

## Service instantiation

```python
from capabilities.fin.glr.general_ledger.service import GeneralLedgerService

svc = GeneralLedgerService(tenant_id="acme", user_id="alice")
```

The service stores all state in plain `dict` stores. Production deployments swap these with repository objects that expose the same dict interface.

### Store attributes

| Attribute | Contents |
|---|---|
| `accounts` | Chart of accounts records |
| `periods` | Accounting period records |
| `journal_entries` | Journal header records |
| `postings` | Posted ledger entries |
| `journal_batches` | Batch grouping records |
| `reconciliations` | Reconciliation header records |
| `budgets` | Budget line records |
| `currency_rates` | FX rate records |
| `recurring_templates` | Recurring journal templates |
| `allocations` | Cost allocation records |
| `agents` | GLR agent registrations |
| `fiscal_years` | Closed fiscal year records |
| `_audit_events` | Immutable audit log |

---

## Key async methods

### Chart of accounts

```python
# Create account
acct = svc.create_account(
    account_id="cash-1", tenant_id="acme", code="1000",
    name="Cash", account_type="asset",
)

# Async v2 with full validation
acct = await svc.create_account_v2(
    tenant_id="acme", account_code="1000", account_name="Cash",
    account_type="asset", parent_code=None, currency="USD",
)

# List
accounts = await svc.chart_of_accounts("acme", include_inactive=False)

# Hierarchy
tree = await svc.account_hierarchy("acme")
```

### Period lifecycle

```python
await svc.open_period_v2("acme", "2026-01", opened_by="controller")
checklist = await svc.period_end_checklist("acme", "2026-01")
await svc.close_period("acme", "2026-01", closed_by="controller")
await svc.lock_period("acme", "2026-01", locked_by="cfo")
await svc.reopen_period("acme", "2026-01", reason="...", authorised_by="cfo")
```

### Journal entry — recommended path

`post_journal_v2` is the one-step create+validate+post path:

```python
posting = await svc.post_journal_v2(
    tenant_id="acme",
    journal_date="2026-01-15",
    journal_type="standard",          # standard | adjustment | accrual | reversal …
    lines=[
        {"account_id": "acct-cash",    "debit": "1000", "credit": "0",
         "description": "Sales receipt", "cost_center": "CC01"},
        {"account_id": "acct-revenue", "debit": "0", "credit": "1000",
         "description": "Service revenue"},
    ],
    description="Invoice INV-001 receipt",
    reference="INV-001",
    posted_by="alice",
)
```

### Reversal

```python
reversal = await svc.reverse_journal_v2(
    tenant_id="acme",
    journal_id="journal-id",
    reversal_date="2026-02-01",
    reversal_description="Accrual reversal",
    reversed_by="bob",
)
```

### Financial statements

```python
tb  = await svc.trial_balance("acme", "2026-01")
bs  = await svc.balance_sheet("acme", "2026-01", comparative_period="2025-01")
inc = await svc.income_statement("acme", "2026-01")
cfs = await svc.cash_flow_statement("acme", "2026-01", method="indirect")
bva = await svc.budget_vs_actual("acme", "2026-01")
pack = await svc.management_accounts_pack("acme", "2026-01")
```

### Year-end close

```python
await svc.year_end_close("acme", fiscal_year=2026, retained_earnings_account="3100")
await svc.opening_balances_new_year("acme", new_fiscal_year=2027)
```

---

## Domain rules

All business rules live in `domain/rules.py` as standalone callable functions that raise `RuleViolation`. They are invoked from `service.py` and are independently testable.

```python
from capabilities.fin.glr.general_ledger.domain.rules import (
    assert_journal_balanced, assert_segregation_of_duties, RuleViolation
)

try:
    assert_journal_balanced(total_debit, total_credit)
    assert_segregation_of_duties(prepared_by="alice", posted_by="alice")
except RuleViolation as e:
    print(e.rule_name, e.reason, e.required_action)
```

---

## Domain calculations

Pure math in `domain/calculations.py`:

```python
from capabilities.fin.glr.general_ledger.domain.calculations import (
    net_balance, functional_amount, revaluation_gain_loss,
    hyperinflationary_restatement, calculate_ratios,
    vat_exclusive_to_inclusive,
)
```

All functions are pure (no I/O), fully type-annotated, and handle edge cases explicitly.

---

## REST API

Base path: `/api/glr`

### Blueprint registration

```python
from capabilities.fin.glr.general_ledger.api import bp
app.register_blueprint(bp)  # auto url_prefix=/api/glr
```

### Tenant identification

Every request must identify the tenant via one of:
- `X-Tenant-ID` header
- `tenant_id` query param
- `tenant_id` in JSON body

### Response envelope

```json
{
    "data": { ... },
    "error": null,
    "meta": { "total": 100, "page": 1, "page_size": 50 }
}
```

Error response:

```json
{
    "data": null,
    "error": { "message": "journal_not_balanced: debits=1000 credits=900", "code": "bad_request" }
}
```

---

## Database schema

Full PostgreSQL DDL in `database/schema.sql`. Key tables:

| Table | Description |
|---|---|
| `gl_tenant` | Tenant configuration, base/functional currency |
| `gl_account` | Chart of accounts, parent-child hierarchy |
| `gl_period` | Accounting periods with status lifecycle |
| `gl_journal_entry` | Journal headers with approval workflow |
| `gl_journal_line` | Debit/credit lines, multi-currency |
| `gl_posting` | Immutable ledger postings |
| `gl_budget` | Budget lines by account/period |
| `gl_reconciliation` | Reconciliation headers |
| `gl_reconciliation_item` | Individual reconciling items |
| `gl_currency_rate` | FX exchange rates |
| `gl_recurring_template` | Recurring journal templates |
| `gl_allocation` | Cost allocation rules |
| `gl_closing_entry` | Year-end closing journal references |

All monetary amounts: `NUMERIC(18,4)`. All IDs: `TEXT` (UUID-7 strings).

---

## Testing

```bash
# All tests
uv run pytest -vxs capabilities/fin/glr/general_ledger/tests/

# Specific suites
uv run pytest capabilities/fin/glr/general_ledger/tests/test_models.py
uv run pytest capabilities/fin/glr/general_ledger/tests/test_service.py
uv run pytest capabilities/fin/glr/general_ledger/tests/test_rules.py
uv run pytest capabilities/fin/glr/general_ledger/tests/test_calculations.py
uv run pytest capabilities/fin/glr/general_ledger/tests/test_api.py

# Type checking
uv run pyright capabilities/fin/glr/general_ledger/
```

No mocks (except LLM). Tests use real in-memory service instances. API tests use Flask test client.

---

## Event streaming

Every state change emits a structured event to `apg.fin.glr.lifecycle` via Bytewax:

```python
{
    "tenant_id": "acme",
    "event_type": "journal_posted",
    "record_id": "journal-abc123",
    "record_type": "ledger_posting",
    "status": "posted",
    "stream": "apg.fin.glr.lifecycle",
    "processor": "bytewax",
    "emitted_at": "2026-01-15T10:30:00Z"
}
```

Downstream capabilities (AP, AR, CBM) subscribe to these events for subledger reconciliation.

---

## Adding a new service method

1. Add the rule assertion(s) to `domain/rules.py`
2. Implement the async method in `service.py` with `_log_*` logging
3. Add the REST endpoint to `api.py`
4. Add screen model function to `views.py`
5. Write tests in `tests/test_service.py` and `tests/test_api.py`

---

## Performance notes

- `chart_of_accounts` iterates `self.accounts` — use an index dict in production
- Trial balance aggregates all postings — add period-scoped materialized views
- `management_accounts_pack` calls 5 reports sequentially — parallelize with `asyncio.gather` in production
- Budget vs actual joins budgets to trial balance — index on `(tenant_id, account_code, period_code)`
