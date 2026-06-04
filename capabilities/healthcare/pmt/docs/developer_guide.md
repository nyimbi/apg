# Patient Management — Developer Guide

**Capability:** `healthcare_pmt` | **Version:** 1.0.0

---

## Architecture

PMT follows APG's layered capability architecture:

```
api.py          ← Flask Blueprint REST API (sync wrappers over async service)
views.py        ← View model builders for UI screens
service.py      ← Async business logic, tenant-scoped, in-memory or PostgreSQL
capability_contract.py  ← Rule engine, configuration, APG composition contract
models.py       ← Pydantic v2 models (all entities, create/update/response)
domain/
  rules.py      ← Pure deterministic rule functions (RuleViolation exceptions)
  calculations.py ← Pure financial and clinical calculations
  adapters.py   ← Protocol adapters for auth, audit, notify, workflow
  events.py     ← Domain event type registry
database/
  store.py      ← InMemoryStore + PostgreSQLStore (Store protocol)
  schema.sql    ← Complete PostgreSQL DDL
```

---

## Coding Standards

All code follows APG CLAUDE.md standards:

- Python 3.12+, async throughout
- **Tabs** (not spaces) for indentation
- Modern typing: `str | None`, `list[str]`, `dict[str, Any]`
- Pydantic v2: `ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)`
- IDs: `str = Field(default_factory=uuid7str)` using `uuid6.uuid7`
- `_log_*` prefixed methods for all console logging
- Runtime `assert` at function start/end
- Domain events emitted after every state change

---

## Service Layer

```python
from apg_healthcare_pmt.service import PatientManagementService
from apg_healthcare_pmt.models import PatientCreate

svc = PatientManagementService()

# Register a patient
patient = await svc.register_patient(PatientCreate(
    tenant_id="nairobi_hosp",
    first_name="Amina", last_name="Odhiambo",
    date_of_birth=datetime(1988, 3, 15),
    gender_code="female", phone="0712345678",
    created_by="reg_001",
))

# Admit
admission = await svc.admit_patient(AdmissionCreate(
    tenant_id="nairobi_hosp",
    patient_id=patient.id,
    admission_type="emergency",
    admitting_provider_id="dr_001",
    attending_provider_id="dr_002",
    unit_id="ED", bed_id=bed.id,
    chief_complaint="Chest pain",
    created_by="nurse_001",
))

# Discharge
discharged = await svc.discharge_patient(
    "nairobi_hosp", admission.id,
    disposition="home",
    physician_order_present=True,
)
```

### Service Constructor

```python
svc = PatientManagementService()
```

The service uses in-memory stores by default. For production PostgreSQL, wire the `store` via the `database/store.py` `get_store(db_url)` factory.

### Tenant Isolation

Every method takes `tenant_id` as its first argument (or via a Pydantic payload). The `_enforce()` method calls `evaluate_capability_rules()` which rejects cross-tenant access at the rule engine level.

---

## Rule Engine

Rules are evaluated in `_enforce()`:

```python
def _enforce(self, context: dict[str, Any]) -> None:
    result = evaluate_capability_rules(context)
    if result["decision"] == "deny":
        raise PolicyViolationError(result["reason"])
```

Add new rules to `capability_contract.py` `RULES` list:

```python
{
    "name": "my_new_rule",
    "condition": {"operation": "my_op", "my_flag": False},
    "effect": {"decision": "deny", "reason": "my_reason", "required_action": "my_action"},
}
```

---

## Domain Rules

All business rules live in `domain/rules.py` as pure functions. No I/O, no side effects.

```python
from apg_healthcare_pmt.domain.rules import assert_physician_discharge_order, RuleViolation

try:
    assert_physician_discharge_order(physician_order_present=False)
except RuleViolation as e:
    print(e.rule_name)        # "discharge_requires_physician_order"
    print(e.required_action)  # "obtain_physician_discharge_order"
```

---

## Domain Calculations

All financial and clinical calculations live in `domain/calculations.py`. All functions are pure.

```python
from apg_healthcare_pmt.domain.calculations import (
    calculate_nhif_benefit,
    calculate_early_warning_score,
    calculate_no_show_risk,
)

# NHIF benefit
benefit = calculate_nhif_benefit("emergency", los_days=3, ward_category="general")
# → Decimal("7500.00")

# NEWS2-style EWS
score, level = calculate_early_warning_score({
    "bp_systolic": 85, "respiratory_rate": 28,
    "spo2": 91, "heart_rate": 130,
    "temperature_c": 38.8, "avpu_score": 1.0,
})
# → (8, "critical")

# No-show risk
risk = calculate_no_show_risk(
    prior_no_shows=2, prior_cancellations=1,
    total_appointments=10, days_until_appointment=14,
    telehealth=False,
)
```

---

## Capability Contract

The contract is the single source of truth for rules, configuration, UI routes, and APG composition registration.

```python
from apg_healthcare_pmt import get_capability_contract, evaluate_capability_rules

contract = get_capability_contract(tenant_id="my_org")
# contract["provides"]  → list of workflow identifiers
# contract["requires"]  → list of adapter capabilities
# contract["rule_engine"]["rules"]  → all governance rules
# contract["ui"]["routes"]  → screen route definitions

result = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation_type": "write",
    "policy_attached": True,
})
# → {"decision": "allow", "actions": [], ...}
```

---

## Adding a New Endpoint

1. Add the Pydantic models to `models.py`
2. Add the business logic to `service.py` as an `async` method
3. Add domain rules to `domain/rules.py`
4. Add the Flask route to `api.py`
5. Add tests to `tests/test_service.py` and `tests/test_models.py`

Example service method skeleton:

```python
async def my_new_operation(
    self,
    tenant_id: str,
    entity_id: str,
    actor_id: str,
) -> dict[str, Any]:
    """Brief description of what this does."""
    assert bool(tenant_id), "tenant_id required"
    assert bool(entity_id), "entity_id required"

    self._enforce({
        "tenant_context_present": bool(tenant_id),
        "operation_type": "write",
        "policy_attached": True,
        "operation": "my_new_operation",
    })

    # ... business logic ...

    result_id = uuid7str()
    self._audit(tenant_id, "my_operation_completed", result_id)
    _log_op("my_new_operation", tenant_id, result_id)
    return {"id": result_id, "status": "completed"}
```

---

## Testing

```bash
# Run all tests
uv run pytest -vxs tests/

# Run specific module
uv run pytest tests/test_service.py -v

# With coverage
uv run pytest tests/ --cov=pmt --cov-report=term-missing
```

Tests use:
- Plain `asyncio.get_event_loop().run_until_complete()` — no `@pytest.mark.asyncio`
- Real objects and real service instances — no mocks
- pytest fixtures for shared setup

---

## Adapters (Platform Integration)

When running inside the APG platform, wire real adapters:

```python
from apg_common_auth import AuthService
from apg_healthcare_pmt.domain.adapters import get_auth_adapter

auth = get_auth_adapter(AuthService.from_env())
# NullAuthAdapter used automatically in standalone mode
```

Adapter protocols:
- `AuthAdapter` — token verification, permission checking
- `AuditAdapter` — structured audit event logging
- `NotifyAdapter` — multi-channel notifications
- `WorkflowAdapter` — workflow engine integration

---

## APG Composition Registration

The entry point `apg.capabilities` group in `pyproject.toml` registers this capability:

```toml
[project.entry-points."apg.capabilities"]
healthcare_pmt = "apg_healthcare_pmt:get_capability_contract"
```

The composition engine discovers it via:

```python
import importlib.metadata
eps = importlib.metadata.entry_points(group="apg.capabilities")
contract = eps["healthcare_pmt"].load()("my_tenant")
```

---

## Database Schema

See `database/schema.sql` for the complete PostgreSQL DDL with:
- All tables with UUID primary keys
- Composite indexes on `(tenant_id, id)` for all tables
- Soft-delete pattern (`is_deleted` + `deleted_at`)
- Audit columns on every table
- Foreign keys with `ON DELETE RESTRICT`

---

## Performance Notes

- All in-memory stores use `dict[tuple[str, str], Model]` keyed by `(tenant_id, id)` for O(1) lookup
- List operations are O(n) and suitable for single-tenant workloads up to ~100k records
- For multi-tenant production, use the PostgreSQL store with connection pooling (asyncpg)
- The NEWS2 EWS calculation is O(1) — safe to call on every vitals record event

© 2025 Datacraft | www.datacraft.co.ke
