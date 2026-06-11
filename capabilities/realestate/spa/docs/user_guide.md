# Space Planning & Management — User Guide

**Capability ID**: `realestate_spa` | **Domain**: `realestate` | **Version**: `1.1.0`

---

## Description

Comprehensive workplace and space management: versioned floor plans, space allocation and deallocation, move management with headcount-threshold approvals, conflict-checked space bookings, anonymised sensor-data ingestion for occupancy analytics, workplace density planning, space chargeback, and seven new analytics and workflow methods added in v1.1.

---

## Installation

```bash
pip install apg-realestate-spa
```

---

## Provides

- `floor_plan_management`
- `space_allocation_engine`
- `move_management_workflow`
- `occupancy_analytics`
- `workplace_density_planning`
- `space_booking_engine`
- `sensor_integration_bridge`
- `chargeback_space_accounting`
- `accessibility_space_finder` *(v1.1)*
- `allocation_expiry_monitor` *(v1.1)*
- `portfolio_benchmarking` *(v1.1)*
- `overcrowding_detection` *(v1.1)*
- `energy_intensity_analytics` *(v1.1)*
- `space_request_workflow` *(v1.1)*
- `zone_analytics` *(v1.1)*

---

## Requires

| Capability | Purpose |
|-----------|---------|
| `auth` | Move and request approval authority |
| `audl` | Space allocation audit trail |
| `mten` | Tenant isolation |
| `conf` | Density targets and booking limits |
| `ntfy` | Density threshold and expiry alerts |
| `wflo` | Large-move and space-request approval workflows |
| `moni` | Real-time occupancy monitoring |
| `mqeb` | Publish space and overcrowding events |
| `schd` | Scheduled occupancy reporting |

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/spa/dashboard` | `realestate_spa:view` | Overview |
| `/realestate/spa/floor-plans` | `realestate_spa:floor_plans` | Floor Plans |
| `/realestate/spa/spaces` | `realestate_spa:spaces` | Spaces |
| `/realestate/spa/spaces/accessible` | `realestate_spa:spaces` | Spaces |
| `/realestate/spa/allocations` | `realestate_spa:allocations` | Allocation |
| `/realestate/spa/allocations/expiries` | `realestate_spa:allocations` | Allocation |
| `/realestate/spa/moves` | `realestate_spa:moves` | Moves |
| `/realestate/spa/bookings` | `realestate_spa:bookings` | Bookings |
| `/realestate/spa/occupancy` | `realestate_spa:occupancy` | Analytics |
| `/realestate/spa/density` | `realestate_spa:density` | Planning |
| `/realestate/spa/requests` | `realestate_spa:requests` | Requests |
| `/realestate/spa/zones/analytics` | `realestate_spa:analytics` | Analytics |
| `/realestate/spa/benchmark` | `realestate_spa:analytics` | Analytics |
| `/realestate/spa/energy` | `realestate_spa:analytics` | Analytics |

---

## Service Methods

### Core (v1.0)

#### Floor Plans

```python
svc = SpaService(tenant_id="t1", actor_id="nyimbi")

# Upload a versioned floor plan
plan = await svc.upload_floor_plan(FloorPlanCreate(
    tenant_id="t1", property_id="bld-001", floor="G",
    file_format="dwg", file_reference="s3://plans/bld001-G.dwg",
    total_area=Decimal("1200"), created_by="nyimbi",
))

# List floor plans for a building
plans = await svc.list_floor_plans("t1", property_id="bld-001")
```

#### Spaces

```python
space = await svc.create_space(SpaceCreate(
    tenant_id="t1", property_id="bld-001", floor_plan_id=plan.id,
    space_ref="G-101", space_type=SpaceType.meeting_room,
    capacity=8, area=Decimal("24"), created_by="nyimbi",
    amenities=["wheelchair_accessible", "hearing_loop", "video_conferencing"],
))

# Filter by type and status
rooms = await svc.list_spaces("t1", space_type="meeting_room", status="available")
```

#### Allocations

```python
alloc = await svc.allocate_space(SpaceAllocationCreate(
    tenant_id="t1", space_id=space.id,
    allocation_type=AllocationType.permanent,
    department_id="dept-sales", occupant_ids=["emp-001"],
    start_date=date(2026, 7, 1), headcount=1, created_by="nyimbi",
))

# End the allocation
await svc.deallocate_space(alloc.id, "t1")
```

#### Bookings

```python
booking = await svc.create_booking(BookingCreate(
    tenant_id="t1", space_id=space.id,
    booking_type=BookingType.meeting_room,
    booked_by="emp-001", start_datetime=datetime(2026, 7, 1, 9, 0),
    end_datetime=datetime(2026, 7, 1, 10, 30), attendees=6,
    created_by="emp-001",
))
```

#### Occupancy

```python
await svc.ingest_occupancy_data(OccupancyDataCreate(
    tenant_id="t1", space_id=space.id,
    sensor_type=SensorType.occupancy_sensor,
    recorded_at=datetime.utcnow(), occupant_count=5,
    data_anonymised=True, created_by="sensor-gw-1",
))

metrics = await svc.calculate_occupancy_metrics("t1", "bld-001", date(2026, 6, 1), date(2026, 6, 30))
```

#### Chargeback

```python
result = await svc.calculate_chargeback(
    "t1", "bld-001", "2026-Q2",
    rate_per_sqm=Decimal("85.00"),
    occupancy_data_verified=True,
)
# result["chargebacks"] is a per-department breakdown
```

---

### Analytics & Workflow (v1.1)

#### `find_accessible_spaces`

Return spaces matching **all** requested accessibility features.  Useful for
HR, legal, and PSED compliance queries.

```python
spaces = await svc.find_accessible_spaces(
    tenant_id="t1",
    required_features=["wheelchair_accessible", "hearing_loop"],
    property_id="bld-001",
    min_capacity=4,
)
# Each item includes accessibility_features: list[str] for the intersection
```

---

#### `check_allocation_expiries`

Proactively surface allocations expiring within a lookahead window.  Urgency
tiers: `critical` (≤ 7 days), `warning` (≤ 14 days), `notice` (≤ N days).

```python
report = await svc.check_allocation_expiries("t1", lookahead_days=30)
# report["critical"] — count of allocations expiring within 7 days
# report["expiring"] — sorted list with days_remaining and urgency tag
```

Wire the result to `ntfy` to send automated renewal-reminder emails.

---

#### `benchmark_portfolio`

Rank properties within the tenant portfolio for a chosen metric.  Returns
percentile ranks and IQR-based outlier flags.

```python
bench = await svc.benchmark_portfolio("t1", metric="utilisation_rate")
# bench["properties"] is sorted ascending by score with rank and percentile
# items where bench["properties"][i]["outlier"] is True need attention
```

Supported metrics: `utilisation_rate`, `sqm_per_person`, `booking_adherence`, `void_rate`.

---

#### `detect_overuse_events`

Identify occupancy readings where sensor count exceeded `capacity × threshold`.
Severity: `high` (≥ 20 % of readings), `medium` (≥ 5 %), `low` (< 5 %).

```python
overuse = await svc.detect_overuse_events(
    tenant_id="t1",
    space_id=space.id,
    period_days=30,
    overcrowding_threshold=1.1,  # 10% over capacity = overuse
)
# overuse["events"] — timestamped list with excess headcount per reading
# overuse["severity"] — "high" | "medium" | "low"
```

Publish `overuse["events"]` to `mqeb` topic `spa.space.overuse` to trigger
FM deep-clean and HVAC scheduling workflows.

---

#### `calculate_energy_per_occupant`

Join external energy meter data with sensor occupancy to compute kWh/person/day
per space.  Flags spaces with zero occupancy and non-zero attributed energy as
waste candidates.

```python
energy = await svc.calculate_energy_per_occupant(
    tenant_id="t1",
    building_id="bld-001",
    period="2026-Q1",
    total_kwh=48_500.0,
)
# energy["portfolio_kwh_per_person_day"] — headline figure for ESG reporting
# energy["waste_candidate_spaces"] — count of spaces consuming energy with 0 occupancy
# energy["spaces"] — per-space breakdown with area_fraction_pct and attributed_kwh
```

---

#### `submit_space_request` / `approve_space_request`

Structured request → approval → matching chain.  Replaces informal
email/Slack negotiation with an auditable record.

```python
# Department head submits request
req = await svc.submit_space_request(
    tenant_id="t1",
    requestor_id="emp-101",
    department_id="dept-engineering",
    requested_space_type="open_plan",
    required_capacity=15,
    required_from=date(2026, 9, 1),
    justification="New hire cohort Q3 2026 — 15 FTE joining 01-Sep",
    preferred_building_id="bld-001",
)

# Space manager approves and receives matching available spaces
approved = await svc.approve_space_request(
    tenant_id="t1",
    request_id=req["id"],
    reviewer_id="space-mgr-001",
    notes="Approved. Three options attached.",
)
# approved["matching_spaces"] — up to 10 available spaces meeting requirements
```

---

#### `get_zone_analytics`

Aggregate metrics for an ad-hoc grouping of spaces (floor wing, neighbourhood,
executive suite) without requiring a persisted zone record.

```python
zone = await svc.get_zone_analytics(
    tenant_id="t1",
    zone_space_ids=["s-001", "s-002", "s-003", "s-004"],
    zone_name="East Wing — Sales",
)
# zone["utilisation_pct"] — occupied fraction
# zone["sqm_per_person"] — current density
# zone["density_ok"] — True if >= 8.0 sqm/person (RICS minimum)
# zone["space_breakdown"] — per-space detail with sensor averages
```

---

## Configuration

All keys are tenant-scoped.  Set via the `conf` capability or environment
variables prefixed `REALESTATE_SPA_`.

| Key | Default | Description |
|-----|---------|-------------|
| `moves.large_move_headcount_threshold` | 20 | Headcount requiring management approval |
| `bookings.max_advance_booking_days` | 90 | Maximum advance booking window |
| `occupancy.anonymise_sensor_data` | true | Mandate anonymisation before ingestion |
| `energy.waste_threshold_kwh` | 1.0 | Minimum attributed kWh to flag a zero-occupancy space |
| `expiry.default_lookahead_days` | 30 | Default expiry scan window |

---

## Interoperability

Reference this capability in `.apg` source files:

```apg
use realestate_spa;
```

Key composition points:

- Floor plans link to `realestate_prm` property and unit records
- Chargeback feeds into `realestate_acc` cost allocation runs
- Move completions may trigger `realestate_prm` unit status updates
- Occupancy analytics inform `realestate_val` DCF rental growth assumptions
- Overcrowding events publish to `mqeb` for FM maintenance dispatch
- Allocation expiry events route through `ntfy` for department-head alerts
- Energy intensity data feeds `esg` sustainability reporting
- Zone analytics compose with `org` hierarchy for budget attribution

---

## Further Reading

- `service.py` — Business logic implementation (all 35+ async methods)
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints (Flask-AppBuilder blueprint)
- `views.py` — FAB views and SQLAlchemy models
- `README.md` — Quick reference including all API routes
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised improvement proposals
- `cap_spec.md` — Formal capability specification
