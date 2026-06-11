# Property Marketing & Management

**Capability ID**: `realestate_prm` | **Domain**: `realestate` | **Version**: `1.1.0`

## Description

Central portfolio management and property marketing platform for all real estate assets.
Registers properties and units, manages owner entities and their distributions, tracks
performance KPIs (occupancy, WAULT, yield), coordinates handovers, and drives the full
marketing funnel — from listing publication and virtual tour orchestration through to
lead capture, urgency scoring, agent routing, and conversion tracking.

---

## Installation

```bash
pip install apg-realestate-prm
```

---

## Provides

| Service | Description |
|---------|-------------|
| `property_portfolio_management` | Register and lifecycle-manage properties across types and tiers |
| `unit_management` | Create, assign, and track individual lettable units |
| `owner_portal_service` | Secure portal with statements, documents, and distribution history |
| `property_performance_reporting` | KPI engine: occupancy, void rate, yield, CAPEX ratio |
| `portfolio_analytics` | Portfolio-wide benchmarking and trend analysis |
| `handover_management` | Structured landlord/developer/contractor handover workflow |
| `owner_distribution_management` | Dual-control net distribution payments |
| `listing_management` | Compose, publish, and multi-channel syndicate property listings |
| `lead_management` | Capture, score, route, and pipeline-track prospect leads |
| `virtual_tour_management` | Register and analytics-track 360°/video virtual tours |
| `marketing_funnel_analytics` | Funnel reporting: listing views → leads → viewings → conversions |

---

## Requires

| Capability | Reason |
|-----------|--------|
| `auth` | User identity and permissions |
| `audl` | Audit trail for status changes and distributions |
| `mten` | Tenant isolation for portfolio data |
| `conf` | Portfolio-level configuration |
| `ntfy` | Void unit alerts, occupancy threshold alerts, lead SLA alerts |
| `wflo` | Board approval workflow for deletions |
| `moni` | Occupancy threshold monitoring |
| `mqeb` | Publish unit/listing status changes to event stream |
| `srch` | Full-text property search |

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/prm/dashboard` | `realestate_prm:view` | Overview |
| `/realestate/prm/portfolio` | `realestate_prm:portfolio` | Portfolio |
| `/realestate/prm/properties` | `realestate_prm:properties` | Properties |
| `/realestate/prm/properties/<id>` | `realestate_prm:properties` | Properties |
| `/realestate/prm/units` | `realestate_prm:units` | Units |
| `/realestate/prm/owners` | `realestate_prm:owners` | Owners |
| `/realestate/prm/owner-portal` | `realestate_prm:owner_portal` | Owners |
| `/realestate/prm/performance` | `realestate_prm:performance` | Analytics |
| `/realestate/prm/listings` | `realestate_prm:listings` | Marketing |
| `/realestate/prm/leads` | `realestate_prm:leads` | Marketing |
| `/realestate/prm/tours` | `realestate_prm:tours` | Marketing |
| `/realestate/prm/marketing/funnel` | `realestate_prm:analytics` | Marketing |

---

## Key Service Methods

### Portfolio & Property

```python
svc = PrmService(tenant_id="acme", actor_id="agent-1")

# Register an owner
owner = await svc.register_owner(OwnerCreate(...))

# Register a property
prop = await svc.register_property(PropertyCreate(...))

# Create a unit inside the property
unit = await svc.create_unit(UnitCreate(property_id=prop.id, ...))

# Portfolio summary
summary = await svc.get_portfolio_summary(tenant_id="acme")

# KPI calculation
kpis = await svc.calculate_kpis(KpiCalculationRequest(
    tenant_id="acme",
    kpi_names=["occupancy_rate", "void_rate"],
    period="2026-06",
    requested_by="agent-1",
))

# Portfolio-wide analytics
report = await svc.property_analytics(period="2026-06", tenant_id="acme")
```

### Owner Distributions

```python
dist = await svc.create_distribution(DistributionCreate(...))

approved = await svc.approve_distribution(
    dist.id, tenant_id="acme",
    approver="alice", second_approver="bob",   # must be distinct
)
```

### Handovers

```python
handover = await svc.create_handover(HandoverCreate(...))
completed = await svc.complete_handover(handover.id, tenant_id="acme")
```

### Inspections & Utilities

```python
inspection = await svc.property_inspection(
    property_id=prop.id,
    inspection_date=date.today(),
    findings=[{"area": "roof", "severity": "critical", "action_required": True}],
    inspector_id="insp-99",
    tenant_id="acme",
)

reading = await svc.utility_management(
    property_id=prop.id,
    utility_type="electricity",
    reading=12450.0,
    previous_reading=11900.0,
    period="2026-06",
    tenant_id="acme",
)
```

---

## Marketing Funnel — Quick Start

### 1. Publish a Listing

```python
listing = await svc.publish_listing(
    tenant_id="acme",
    property_id=prop.id,
    unit_id=unit.id,
    headline="Grade-A Office Suite — Westlands Nairobi",
    description="Open-plan 450 sqm floor with panoramic city views...",
    asking_rent=220000.0,
    rent_frequency="monthly",
    available_from=date(2026, 7, 1),
    media_urls=["https://cdn.example.com/img/prop001_1.jpg"],
    channels=["website", "portal"],
    actor_id="agent-1",
)
```

### 2. Create a Virtual Tour

```python
tour = await svc.create_virtual_tour(
    tenant_id="acme",
    property_id=prop.id,
    unit_id=unit.id,
    tour_name="Westlands Suite — 360 Tour",
    media_type="360_images",
    scene_urls=[
        "https://cdn.example.com/tours/prop001/reception.jpg",
        "https://cdn.example.com/tours/prop001/boardroom.jpg",
    ],
    floor_plan_url="https://cdn.example.com/tours/prop001/floorplan.png",
    actor_id="agent-1",
)
# shareable_url is auto-generated and auto-linked to the active listing
print(tour["shareable_url"])   # /realestate/prm/tours/<id>/view
```

### 3. Capture a Lead

```python
lead = await svc.capture_lead(
    tenant_id="acme",
    property_id=prop.id,
    contact_name="Sarah Kamau",
    contact_email="sarah@example.com",
    source="virtual_tour",
    budget_min=200000.0,
    budget_max=250000.0,
    message="Ready to move in ASAP — need 450 sqm.",
    actor_id="portal",
)
print(lead["urgency_score"])   # 7 — virtual_tour source + budget + "ASAP"
```

### 4. Record a Tour View (boosts lead urgency)

```python
event = await svc.record_tour_view(
    tour_id=tour["id"],
    tenant_id="acme",
    viewer_session_id="sess-abc123",
    dwell_seconds=245,
    lead_id=lead["id"],
)
```

### 5. Assign and Advance the Lead

```python
lead = await svc.assign_lead(lead["id"], tenant_id="acme", agent_id="agent-1")

lead = await svc.update_lead_status(
    lead["id"], tenant_id="acme", new_status="viewing",
    notes="Viewing scheduled 2026-06-15 10:00",
)
lead = await svc.update_lead_status(
    lead["id"], tenant_id="acme", new_status="converted",
    notes="Lease heads-of-terms agreed",
)
```

### 6. Marketing Funnel Report

```python
funnel = await svc.marketing_funnel_report(
    tenant_id="acme",
    property_id=prop.id,
    period="2026-06",
)
# {
#   "active_listings": 1,
#   "total_views": 124,
#   "total_leads": 18,
#   "viewings": 7,
#   "offers": 3,
#   "conversions": 2,
#   "lead_to_view_rate_pct": 14.52,
#   "view_to_viewing_rate_pct": 38.89,
#   "viewing_to_offer_rate_pct": 42.86,
#   "offer_to_conversion_rate_pct": 66.67,
#   "overall_conversion_rate_pct": 11.11,
#   ...
# }
```

### 7. Unpublish a Listing

```python
await svc.unpublish_listing(
    listing["id"], tenant_id="acme", reason="unit_let"
)
```

---

## Lead Pipeline Statuses

| Status | Meaning |
|--------|---------|
| `new` | Captured, not yet reviewed |
| `assigned` | Allocated to an agent |
| `viewing` | Property viewing booked or completed |
| `offer` | Offer made, heads-of-terms in progress |
| `converted` | Lease or sale agreed — feeds `realestate_lea` |
| `lost` | Prospect withdrew or went elsewhere |
| `on_hold` | Paused at prospect's request |

---

## Lead Urgency Scoring

Urgency score is computed at capture and updated on tour engagement:

| Signal | Points |
|--------|--------|
| `budget_min > 0` | +1 |
| `budget_max > 0` | +1 |
| source = `virtual_tour` | +2 |
| message contains urgency keyword | +3 |
| Each `record_tour_view` call with `lead_id` | +2 |

Leads are returned sorted by urgency score descending from `list_leads()`.

---

## Interoperability

`realestate_prm` is the anchor capability for the real estate domain.  Reference it in `.apg` source files:

```apg
use realestate_prm;
```

Downstream capabilities that depend on `property_id`:
- `realestate_lea` — lease drafting triggered by lead conversion
- `realestate_ren` — rent collection linked to unit occupancy
- `realestate_mai` — maintenance work orders per property
- `realestate_val` — income-based valuations consuming occupancy data
- `realestate_acc` — financial accounts consuming distribution records

---

## Configuration

All configuration keys are tenant-scoped.  Set via the `conf` capability or environment
variables prefixed with `REALESTATE_PRM_`.

| Key | Default | Description |
|-----|---------|-------------|
| `portfolio.supported_tiers` | core, core_plus, … | Valid portfolio tier values |
| `governance.data_room_access_always_logged` | true | Hard gate on data room access |
| `governance.dual_control_for_distributions` | true | Two approvers required for owner payments |
| `marketing.lead_urgency_decay_days` | 14 | Days without contact before urgency decays |
| `marketing.listing_channels` | website | Default channels for new listings |

---

## Further Reading

- `service.py` — All business logic; primary reference for available methods
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoint definitions
- `views.py` — Flask-AppBuilder Blueprint views
- `README.md` — Quick reference and composability notes
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 planned enhancements
- `tests/test_service.py` — Comprehensive service tests
