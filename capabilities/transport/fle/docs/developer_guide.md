# Fleet Management — Developer Guide

**Capability**: `transport_fle` v2.0.0  
**Stack**: Python 3.12+, Pydantic v2, Flask Blueprint, PostgreSQL 14+  
**Audience**: Backend engineers integrating or extending the FLE capability

---

## Architecture

```
capabilities/transport/fle/
├── __init__.py              # Public API, register_capability()
├── models.py                # Pydantic v2 models — enums, entities, reports
├── service.py               # FleetService — all async business logic
├── api.py                   # Flask Blueprint: /api/fle/v1/*
├── views.py                 # Flask Blueprint: /fle/*
├── domain/
│   ├── rules.py             # Pure business rule functions + RuleViolation
│   ├── calculations.py      # Pure financial/operational calculations
│   └── adapters.py          # Protocol adapters (auth, audit, notify, workflow)
├── database/
│   └── schema.sql           # Complete PostgreSQL schema
└── tests/
    ├── test_models.py        # Pydantic validation tests
    ├── test_service.py       # Service integration tests (in-memory)
    └── test_domain.py        # Rules & calculations unit tests
```

---

## Service Layer

```python
from capabilities.transport.fle import FleetService, VehicleCreate

svc = FleetService(db_session, tenant_id="acme", actor_id="user-123")
vehicle = await svc.register_vehicle(VehicleCreate(
    tenant_id="acme",
    vehicle_type="rigid_truck",
    registration="KCA 001T",
    vin="WAUZZZ8K9BA123456",
    make="Mercedes", model="Actros", year=2022,
    fuel_type="diesel", ownership_type="owned",
))
```

### Constructor

```python
FleetService(db_session: Any, tenant_id: str, actor_id: str)
```

- `db_session`: SQLAlchemy async session OR any object (for in-memory/test mode the service uses `_InMemoryDB`)
- `tenant_id`: enforced on every read/write operation
- `actor_id`: logged on every domain event

### In-Memory Mode (Testing)

```python
class _DB: pass
svc = FleetService(_DB(), "tenant_test", "actor_test")
```

No database required. All data stored in `db._fle_store`.

### Domain Events

Every state change emits an event via `_emit_event()`:

```python
svc._events  # list[dict] — all events this session
# e.g. {"event_type": "vehicle.registered", "entity_id": "...", "tenant_id": "..."}
```

In production, wire this to APG's messaging infrastructure (Bytewax/Bytewax/APG mqeb).

---

## Models

All models use:
```python
model_config = ConfigDict(
    extra="forbid",
    validate_by_name=True,
    validate_by_alias=True,
    populate_by_name=True,
)
```

### ID generation

```python
from capabilities.transport.fle.models import uuid7str
id = uuid7str()  # UUID7 — time-sortable, unique
```

### Pattern: Create / Response

- `XxxCreate` — input validation for new records (no id/timestamps)
- `XxxUpdate` — partial update (all fields optional)
- `XxxResponse(FleBase)` — full response with id, tenant_id, created_at, updated_at, is_deleted

---

## Domain Rules

All business rules live in `domain/rules.py` as pure functions:

```python
from capabilities.transport.fle.domain.rules import (
    assert_vehicle_active_for_dispatch,
    RuleViolation,
)

try:
    assert_vehicle_active_for_dispatch(vehicle.status.value)
except RuleViolation as e:
    print(e.rule, e.message)  # VEH-005, "Vehicle status 'in_maintenance' ..."
```

Rule codes:
- `VEH-001..009` — vehicle rules
- `DRV-001..005` — driver rules
- `TACHO-001..005` — EU tachograph (EC 561/2006)
- `HOS-001..003` — US HOS (49 CFR 395)
- `TRIP-001..005` — trip rules
- `OVL-001` — axle overload
- `FUEL-001`, `ODO-001` — fuel/odometer
- `INC-001..002` — incident rules
- `HIRE-001..002` — hire/rental rules
- `MNT-001` — maintenance rules

---

## Calculations

All in `domain/calculations.py` — pure, type-safe, Decimal for money:

```python
from capabilities.transport.fle.domain.calculations import (
    calculate_fuel_cost,
    calculate_fuel_efficiency_l100km,
    calculate_tco,
    calculate_driver_score,
    predict_oil_change_due,
)

fuel_cost = calculate_fuel_cost(Decimal("120"), Decimal("185"))  # → 22200.00
eff = calculate_fuel_efficiency_l100km(Decimal("80"), Decimal("400"))  # → 20.00
score = calculate_driver_score(2, 1, 0, 0, 1, 0, 0, Decimal("500"))
```

---

## REST API

Base URL: `/api/fle/v1`

Required headers:
```
X-Tenant-ID: <tenant_id>
X-Actor-ID: <user_id>
```

### Standard patterns

```
GET    /<resource>?page=1&per_page=50&<filters>   → paginated list
POST   /<resource>                                  → create (201)
GET    /<resource>/<id>                             → detail
PUT    /<resource>/<id>                             → partial update
DELETE /<resource>/<id>                             → soft delete
POST   /<resource>/<id>/<action>                    → state transition
GET    /reports/<type>                              → report
GET    /dashboard                                   → KPIs
GET    /health                                      → health check
```

### Error responses

```json
{"error": "Vehicle KCA 001T not found", "code": "not_found"}
```

HTTP status codes: 200 OK, 201 Created, 400 Bad Request, 403 Forbidden, 404 Not Found, 422 Unprocessable Entity, 500 Internal Error.

---

## Flask Registration

```python
from flask import Flask
from capabilities.transport.fle import register_capability

app = Flask(__name__)
register_capability(app)
# Registers /api/fle/v1/* and /fle/* blueprints
```

Or register blueprints individually:

```python
from capabilities.transport.fle.api import fle_bp
from capabilities.transport.fle.views import fle_views_bp

app.register_blueprint(fle_bp)
app.register_blueprint(fle_views_bp)
```

---

## Database

Apply schema:
```bash
psql $DATABASE_URL -f capabilities/transport/fle/database/schema.sql
```

Key design decisions:
- All tables use `TEXT` primary keys (UUID7 strings — time-sortable)
- Tenant isolation via `tenant_id TEXT NOT NULL` on every table
- Soft deletes via `is_deleted BOOLEAN NOT NULL DEFAULT FALSE`
- Partial indexes on `WHERE NOT is_deleted` for all list queries
- `fle_trips`, `fle_tachograph_records`, `fle_telematics_events`, `fle_domain_events` are range-partitioned by date for scalability
- `fle_telematics_events` has GIN index on `payload` JSONB for event-type queries
- Auto-updating `updated_at` via trigger `fle_set_updated_at()`

---

## Testing

```bash
# All tests
python -m pytest capabilities/transport/fle/tests/ -v

# Domain/calculations only
python -m pytest capabilities/transport/fle/tests/test_domain.py -v

# Service integration
python -m pytest capabilities/transport/fle/tests/test_service.py -v

# Model validation
python -m pytest capabilities/transport/fle/tests/test_models.py -v
```

No mocks. No `@pytest.mark.asyncio` decorators.  
Async via `asyncio.get_event_loop().run_until_complete()`.

---

## Extending

### Adding a new entity

1. Add enums/models to `models.py` (Create, Update, Response variants)
2. Add CRUD + business methods to `service.py`
3. Add rules to `domain/rules.py`
4. Add API routes to `api.py`
5. Add UI view to `views.py`
6. Add table to `database/schema.sql`
7. Add tests to `tests/`

### Connecting to APG platform adapters

```python
from capabilities.transport.fle.domain.adapters import get_auth_adapter, get_audit_adapter

auth = get_auth_adapter()   # returns NullAuthAdapter if apg_common_auth not installed
audit = get_audit_adapter() # returns NullAuditAdapter if apg_common_audl not installed
```

When APG platform packages are installed (`apg_common_auth`, `apg_common_audl`, etc.), real adapters are used automatically.

---

## APG Composition Registration

The capability registers under `transport_fle` in APG's composition engine.

**Provides:** vehicle_lifecycle_workflow, driver_management_workflow, fleet_utilisation_analytics_workflow, fleet_compliance_workflow, telematics_integration_workflow

**Requires:** auth, audl, mten, conf, ntfy, wflo, moni, comp, mqeb, schd

See `capability_contract.py` for the full contract definition.
