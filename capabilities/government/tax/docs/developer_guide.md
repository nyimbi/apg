# Tax Administration — Developer Guide

© 2025 Datacraft | Author: Nyimbi Odero

---

## Architecture

```
capabilities/government/tax/
├── __init__.py               # Package metadata, capability_id = "government_tax"
├── models.py                 # Pydantic v2 entities (933 lines)
├── service.py                # TaxAdministrationService — 60+ async-compatible methods
├── api.py                    # Flask Blueprint: /api/v1/tax/* (REST)
├── views.py                  # Flask Blueprint: /tax/* (HTML UI)
├── capability_contract.py    # Rule engine + contract shape
├── domain/
│   ├── rules.py              # Pure business rule assertions (RuleViolation)
│   └── calculations.py       # Pure financial calculations
├── database/
│   └── schema.sql            # Normalized PostgreSQL schema (tax.* namespace)
└── tests/
    ├── conftest.py           # sys.path setup + canonical module pre-loading
    ├── test_models.py        # 40+ Pydantic model unit tests
    ├── test_calculations.py  # 50+ pure calculation tests
    ├── test_rules.py         # 70+ business rule tests
    ├── test_service.py       # 80+ service integration tests
    ├── test_api.py           # 60+ Flask Blueprint API tests
    └── test_contract.py      # Contract shape + rule evaluation tests
```

### Key Design Decisions

- **In-process store**: `_Store(dict)` keyed by `(tenant_id, record_id)`. Replace with SQLAlchemy + async session for production.
- **Tenant isolation**: every store operation scopes by `tenant_id`. Never query across tenants.
- **Dual interface**: legacy positional API preserved via wrapper methods (`file_return`, `raise_assessment` etc.) delegating to the new keyword API.
- **Domain rules separate from service**: `domain/rules.py` contains pure assertion functions; `service.py` calls them via `_enforce()` (contract engine) or directly via `assert`.
- **No Flask-AppBuilder**: plain Flask Blueprint per CLAUDE.local.md instruction.

---

## Coding Standards

Follow `CLAUDE.md` exactly:

```python
# TABS, not spaces
# Modern typing
def example(data: dict[str, Any], ids: list[str]) -> str | None:
    ...

# Pydantic v2 config on every model
class MyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

# UUID7 IDs
id: str = Field(default_factory=uuid7str)

# Runtime assertions at function entry
def my_method(self, value: str) -> None:
    assert value and value.strip(), "value required"
    ...
```

---

## Data Models

All entities inherit from `TaxBase`:

```python
class TaxBase(BaseModel):
    id: str = Field(default_factory=uuid7str)
    tenant_id: NonEmpty
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "system"
    is_deleted: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### Entity Hierarchy

```
TaxpayerResponse
  └── TaxObligationResponse (FK: taxpayer_id)
      └── TaxReturnResponse (FK: taxpayer_id, tax_pin)
          └── TaxAssessmentResponse (FK: return_id, taxpayer_id)
              ├── TaxDebtResponse (FK: assessment_id, taxpayer_id)
              ├── PenaltyResponse (FK: assessment_id)
              ├── InterestResponse (FK: assessment_id)
              └── ObjectionResponse (FK: assessment_id)
                  └── AppealResponse (FK: objection_id)
TaxPaymentResponse (FK: taxpayer_id, optional: assessment_id, return_id)
TaxAuditResponse (FK: taxpayer_id)
  └── AuditFindingResponse (FK: audit_id)
TaxRefundResponse (FK: taxpayer_id, return_id)
TaxClearanceCertificateResponse (FK: taxpayer_id)
EOIRequest (FK: subject_taxpayer_id)
```

---

## Service Layer

### Instantiation

```python
from apg_government_tax.service import TaxAdministrationService

svc = TaxAdministrationService()
# All state is in-process dicts; swap for DB-backed stores in production
```

### Key Method Groups

| Group | Methods |
|-------|---------|
| Registration | `register_taxpayer`, `update_taxpayer`, `deregister_taxpayer`, `taxpayer_search`, `verify_tin` |
| Returns | `submit_return`, `file_nil_return`, `validate_return`, `amend_return`, `return_filing_status`, `filing_history` |
| Assessment | `issue_assessment`, `calculate_penalty_and_interest` |
| Objections | `raise_objection`, `process_objection`, `file_appeal` |
| Audit | `open_audit_case`, `conduct_audit`, `close_audit_case`, `audit_case_analytics` |
| Payments | `process_tax_payment`, `allocate_payment_to_assessments` |
| Debt | `issue_demand_notice`, `debt_collection_action` |
| Refunds | `refund_application`, `verify_refund`, `approve_refund`, `refund_analytics` |
| Clearance | `issue_tax_clearance_certificate` |
| EOI | `exchange_of_information` |
| Reports | `dashboard_summary`, `revenue_collection_report`, `compliance_rate_report`, `delinquency_report` |

### Event Emission

Every state change appends to `svc.audit_events`:

```python
def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
    self.audit_events.append({
        "tenant_id": tenant_id,
        "event_type": event_type,
        "reference_id": reference_id,
        "processor": "bytewax",
        "recorded_at": _now().isoformat(),
    })
```

In production, replace with a Bytewax stream publisher.

---

## REST API

### Blueprint Registration

```python
from flask import Flask
from apg_government_tax.api import tax_bp

app = Flask(__name__)
app.register_blueprint(tax_bp)
```

### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `X-Tenant-ID` | Yes | Tenant scoping. Defaults to `"default"` |
| `X-Actor-ID` | No | Audit actor. Defaults to `"system"` |
| `Content-Type` | Yes (POST/PUT) | `application/json` |

### Response Envelope

```json
{
  "data": { ... },
  "meta": { "total": 100, "limit": 50, "offset": 0 }
}
```

Errors:
```json
{ "error": "taxpayer not found: A123456789B", "code": 404 }
```

### Replacing the Service Instance

For production DI, replace `api._svc`:

```python
import apg_government_tax.api as tax_api
tax_api._svc = MyDBBackedTaxService(db_session, tenant_id)
```

---

## Domain Rules

Business rules in `domain/rules.py` are pure functions:

```python
from apg_government_tax.domain.rules import (
    assert_objection_within_deadline,
    assert_no_outstanding_debt_for_clearance,
    RuleViolation,
)

try:
    assert_objection_within_deadline(assessment_date, objection_date)
except RuleViolation as e:
    print(e.rule_name, e.reason, e.required_action)
```

All rules raise `RuleViolation(rule_name, reason, required_action)`.

---

## Financial Calculations

`domain/calculations.py` — all pure, `Decimal`-based:

```python
from apg_government_tax.domain.calculations import (
    calculate_income_tax,
    calculate_vat_payable,
    calculate_late_filing_penalty,
    calculate_interest,
    calculate_compliance_risk_score,
)

tax = calculate_income_tax(Decimal("1_000_000"))
penalty = calculate_late_filing_penalty(Decimal("200_000"))
score, category = calculate_compliance_risk_score(
    years_registered=3, returns_filed=12, returns_due=12,
    payments_on_time=11, payments_due=12,
    open_audits=0, prior_fraud_flags=0,
    days_avg_late_filing=5.0, debt_to_turnover_ratio=0.02,
)
```

Never use `float` for monetary values — always `Decimal`.

---

## Database

### Schema Namespace

All tables live in the `tax` schema. Apply with:

```bash
psql $DATABASE_URL -f database/schema.sql
```

### Key Tables

| Table | Notes |
|-------|-------|
| `tax.taxpayers` | Unique index on `(tenant_id, tax_pin)` |
| `tax.tax_returns` | Index on `(tenant_id, tax_pin, period)` |
| `tax.tax_assessments` | FK to `tax_returns` |
| `tax.tax_debts` | Generated `total_amount` and `balance` columns |
| `tax.audit_trail` | Immutable append-only, BigSerial PK |
| `tax.compliance_risk_profiles` | Cached/materialized scores |

### Views

| View | Purpose |
|------|---------|
| `tax.v_taxpayer_debt_summary` | Outstanding debt per taxpayer |
| `tax.v_monthly_filing_compliance` | Filing compliance by month |
| `tax.v_revenue_by_tax_type` | Revenue grouped by tax type |
| `tax.v_audit_pipeline` | Audit status counts |

---

## Testing

```bash
# All tests
python -m pytest tests/ -q

# Specific module
python -m pytest tests/test_calculations.py -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

### Test Conventions

- No `@pytest.mark.asyncio` decorators
- Real objects only — no mocks except LLM calls
- `conftest.py` pre-loads canonical module references to prevent `sys.modules` pollution from `_load()` helpers

---

## Extending the Capability

### Adding a New Tax Type

1. Add value to `TaxType` enum in `models.py`
2. Add mapping in `service.py` `_rt_map` dicts
3. Add rate constants if applicable
4. Add calculation function in `domain/calculations.py`
5. Update `SUPPORTED_TAX_TYPES` in `capability_contract.py`
6. Add test in `tests/test_calculations.py`

### Adding a New Service Method

```python
async def my_new_operation(
    self,
    taxpayer_id: str,
    *,
    tenant_id: str = "default",
    created_by: str = "system",
) -> dict[str, Any]:
    """Docstring required."""
    assert taxpayer_id and taxpayer_id.strip(), "taxpayer_id required"
    # ... implementation
    self._audit(tenant_id, "my_new_operation", taxpayer_id)
    return result
```

### APG Composition Registration

The capability is already registered via the `pyproject.toml` entry point:

```toml
[project.entry-points."apg.capabilities"]
government_tax = "apg_government_tax:get_capability_contract"
```

This allows APG's composition engine to discover and orchestrate this capability.

---

## Performance Notes

- The in-process `_Store` is O(n) for tenant scans — replace with indexed DB queries for >10k records per tenant.
- `allocate_payment_to_assessments` sorts all outstanding debts in memory — add `ORDER BY due_date` index in the DB adapter.
- `compliance_rate_report` is O(taxpayers × returns) — materialize into `tax.compliance_risk_profiles` on a schedule.
- `delinquency_report` scans all debts — partition `tax.tax_debts` by `due_date` for large datasets.
