# Emergency Management — User Guide

**Capability ID**: `government_eme` | **Domain**: `government` | **Version**: `2.0.0`
**© 2025 Datacraft** | www.datacraft.co.ke

---

## Description

`government_eme` is the APG capability for end-to-end emergency management operations. It implements the Incident Command System (ICS) and NIMS frameworks with AI-assisted decision support, CAP-compliant public alerting, NATS-backed event sourcing, and cross-capability choreography.

Key operational domains:
- Incident command and lifecycle management
- Resource mobilisation, tracking, and gap prediction
- Multi-agency and inter-jurisdictional coordination
- Emergency Operations Centre (EOC) management
- Evacuation and shelter management
- Relief distribution and casualty tracking
- Damage assessment
- Public alerting (CAP v1.2)
- After-action reviews with SAIR improvement tracking

---

## Installation

```bash
pip install apg-government-eme
```

---

## Quick Start

```python
from capabilities.government.eme.service import EmergencyManagementService

svc = EmergencyManagementService(tenant_id="nairobi_county", actor_id="commander_01")

# Declare an emergency
emergency = svc.declare_emergency(
    type="flood",
    affected_area="Westlands Sub-County",
    severity="critical",
    declared_by="commander_01",
)
incident_id = emergency["id"]

# Activate EOC
eoc = svc.activate_eoc(
    emergency_id=incident_id,
    location="City Hall EOC",
    staff_ids=["ops_001", "comms_002", "log_003"],
)

# Mobilise resources
resources = svc.resource_mobilisation(
    emergency_id=incident_id,
    resources=[
        {"type": "personnel", "quantity": 50, "unit": "persons", "agency": "Kenya Red Cross"},
        {"type": "vehicle", "quantity": 10, "unit": "vehicles", "agency": "Kenya Police"},
    ],
)

# Broadcast public alert (CAP-compliant, async)
import asyncio
alert = asyncio.run(svc.async_broadcast_cap_alert(
    incident_id=incident_id,
    event="Flash Flood",
    urgency="Immediate",
    severity="Extreme",
    certainty="Observed",
    headline="FLOOD ALERT: Evacuate Westlands immediately",
    description="Flash flooding is occurring in Westlands Sub-County. Water levels are rising rapidly.",
    instruction="Move to higher ground now. Do not attempt to drive through floodwater.",
    affected_areas=["Westlands", "Parklands", "Kangemi"],
    channels=["sms", "push", "ussd"],
))
```

---

## Core Workflows

### 1. Incident Declaration and Lifecycle

```python
# Full-parameter declaration (for programmatic integration)
incident = svc.declare_incident(
    incident_id="INC-2025-001",
    tenant_id="nairobi_county",
    incident_type="flood",
    severity="critical",
    location_reference="Westlands, Nairobi",
    commander_id="commander_01",
    description="Major flooding following 200mm rainfall",
    evidence_reference="REPORT-2025-001",
)

# Phase transitions
svc.transition_phase("INC-2025-001", "nairobi_county", "response")
svc.transition_phase("INC-2025-001", "nairobi_county", "recovery")

# Close incident
svc.incident_close("INC-2025-001", "All clear issued", "commander_01")
```

**Supported phases**: `detection` → `notification` → `activation` → `response` → `recovery` → `stand_down` → `after_action`

### 2. EOC Management

```python
eoc = svc.update_eoc(
    eoc_id="EOC-001",
    tenant_id="nairobi_county",
    incident_id="INC-2025-001",
    eoc_status="activated",
    command_structure="ics",
    activation_authority="county_governor",
    evidence_reference="ACTIVATION-ORDER-001",
    authorised=True,
)
```

**EOC activation requires explicit authority.** Attempts without `authorised=True` raise `PermissionError`.

### 3. Resource Management

```python
# Mobilise a resource
svc.mobilise_resource(
    resource_id="RES-001",
    tenant_id="nairobi_county",
    incident_id="INC-2025-001",
    resource_type="personnel",
    quantity=25,
    unit="persons",
    responsible_agency="Kenya Red Cross",
    evidence_reference="DEPLOY-ORDER-001",
)

# Predict resource gaps 4 hours ahead (async)
gaps = asyncio.run(svc.async_predict_resource_gaps("INC-2025-001", horizon_hours=4))
# Returns: gap_analysis, critical_shortages, recommendations

# Track resource position (from AVL feed)
position = asyncio.run(svc.async_update_resource_position(
    resource_id="RES-001",
    latitude=-1.2634,
    longitude=36.8026,
    heading=45.0,
    speed_kmh=60.0,
    status="en_route",
))
# Returns: GeoJSON Feature

# Request mutual aid (async, EMAC format)
aid = asyncio.run(svc.async_submit_mutual_aid_request(
    incident_id="INC-2025-001",
    requesting_agency="Nairobi County",
    aid_type="personnel",
    target_jurisdiction="kiambu_county",
    resources_requested=[{"type": "personnel", "quantity": 100, "unit": "persons"}],
    urgency="immediate",
))
```

### 4. Multi-Agency Coordination

```python
svc.multi_agency_coordination(
    emergency_id="INC-2025-001",
    agencies=[
        {"type": "government", "name": "Kenya Red Cross", "contact": "krc@redcross.or.ke", "role": "relief"},
        {"type": "ngo", "name": "UNHCR Kenya", "contact": "unhcr@ke.un.org", "role": "shelter"},
        {"type": "military", "name": "KDF Engineers", "contact": "kdf@defence.go.ke", "role": "rescue"},
    ],
)
```

### 5. Evacuation and Shelter

```python
# Manage evacuations
evac = svc.evacuation_management(
    emergency_id="INC-2025-001",
    zones=["Zone-A", "Zone-B", "Zone-C"],
)

# Activate a shelter
shelter = svc.shelter_assign(
    incident_id="INC-2025-001",
    shelter_id="SHELTER-001",
    capacity=500,
    location="Westlands Primary School",
)

# Real-time occupancy updates (async)
occupancy = asyncio.run(svc.async_update_shelter_occupancy(
    shelter_id="SHELTER-001",
    incident_id="INC-2025-001",
    check_ins=45,
    check_outs=3,
))
# capacity_alert=True triggers when occupancy >= 90%
```

### 6. Situation Reporting

```python
# Standard SITREP from incident state
sitrep = svc.situation_report("INC-2025-001", "OP-1")

# AI-drafted ICS-209 narrative (async, requires Ollama)
draft = asyncio.run(svc.async_generate_sitrep_narrative(
    incident_id="INC-2025-001",
    period="OP-2",
    model="mistral",
))
# Returns narrative_draft with status="draft_pending_review"
# Falls back to structured template when Ollama is unavailable
```

### 7. Public Alerting (CAP v1.2)

```python
alert = asyncio.run(svc.async_broadcast_cap_alert(
    incident_id="INC-2025-001",
    event="Flash Flood",
    urgency="Immediate",          # CAP urgency values
    severity="Extreme",           # CAP severity values
    certainty="Observed",         # CAP certainty values
    headline="FLOOD ALERT",       # truncated to 160 chars for SMS
    description="...",
    instruction="Evacuate now.",
    affected_areas=["Westlands"],
    channels=["sms", "push", "eas", "ussd"],  # None = all channels
))
```

CAP envelopes are published to NATS subjects `eme.broadcast.{channel}` for channel-specific adapter consumers.

### 8. After-Action Review

```python
# Record AAR
aar = svc.after_action_review(
    emergency_id="INC-2025-001",
    findings=[
        "Communication system failure in first 2 hours — improvement needed",
        "Volunteer coordination was a strength",
        "Critical failure: resource tracking was manual",
    ],
)

# Add SAIR improvement action (async)
action = asyncio.run(svc.async_create_improvement_action(
    aar_id=aar["id"],
    title="Implement digital resource tracking",
    category="improvement",
    description="Deploy AVL-enabled tracking for all mobilised vehicles",
    owner_id="ops_chief_001",
    due_date="2025-09-01",
))
```

### 9. Incident Escalation

```python
# Manual or ML-triggered escalation
escalation = asyncio.run(svc.async_escalate_incident(
    incident_id="INC-2025-001",
    new_severity="catastrophic",
    escalation_reason="Dam breach confirmed, downstream population at risk",
    escalated_by="hydrology_sensor_ml",
))
# Publishes to eme.alerts.escalation.{incident_id}
# Triggers EOC auto-activation if new severity is critical/catastrophic
```

### 10. Cross-Capability Event Choreography

```python
events = asyncio.run(svc.async_publish_cross_capability_events(
    incident_id="INC-2025-001",
    target_capabilities=["government_law", "government_bud", "intel"],
))
# Publishes CloudEvents to apg.{capability}.events.eme_triggered
```

### 11. Incident Timeline Replay

```python
timeline = asyncio.run(svc.async_replay_incident_timeline("INC-2025-001"))
# Returns ordered list of all audit events for the incident
# Production: reads from NATS JetStream stream EME_EVENTS
```

### 12. ICP Common Picture

```python
picture = asyncio.run(svc.async_render_icp_picture(
    incident_id="INC-2025-001",
    include_layers=["resources", "evacuations"],
))
# Returns GeoJSON FeatureCollection for MapLibre GL rendering
# Published to eme.icp.{incident_id}.picture every 60s in production
```

---

## Analytics and Dashboards

```python
# Period analytics
analytics = svc.emergency_analytics("2025-Q2")

# Dashboard summary
dashboard = svc.dashboard_summary("nairobi_county")

# Volunteer matching
matches = asyncio.run(svc.async_match_volunteers(
    incident_id="INC-2025-001",
    required_skills=["medical", "search_and_rescue", "logistics"],
    max_results=50,
))
```

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables:

| Environment Variable | Description |
|---|---|
| `OLLAMA_BASE_URL` | Ollama endpoint for ML severity and SITREP generation |
| `NATS_URL` | NATS JetStream URL for event streaming |
| `GOVERNMENT_EME_MAX_INCIDENTS` | Per-tenant active incident limit |
| `GOVERNMENT_EME_ALERT_CHANNELS` | Comma-separated default alert channels |

---

## NATS Subjects Reference

| Subject | Direction | Description |
|---|---|---|
| `eme.events.{tenant_id}.{incident_id}` | Publish | Durable incident event log |
| `eme.broadcast.sms` | Publish | CAP SMS alert fan-out |
| `eme.broadcast.push` | Publish | CAP push notification fan-out |
| `eme.broadcast.eas` | Publish | CAP Emergency Alert System |
| `eme.broadcast.ussd` | Publish | CAP USSD broadcast |
| `eme.alerts.escalation.{incident_id}` | Publish | Severity escalation events |
| `eme.alerts.resource_gap.{incident_id}` | Publish | Resource shortage predictions |
| `eme.alerts.shelter_capacity.{shelter_id}` | Publish | Shelter over-capacity warnings |
| `eme.avl.{resource_id}` | Subscribe | Inbound GPS telemetry |
| `eme.mutual_aid.outbound.{jurisdiction}` | Publish | EMAC mutual aid requests |
| `eme.icp.{incident_id}.picture` | Publish | Live ICP GeoJSON picture |
| `apg.{capability}.events.eme_triggered` | Publish | Cross-capability CloudEvents |

---

## File Structure

```
capabilities/government/eme/
├── service.py              — Business logic (sync + async methods)
├── models.py               — Dataclass models
├── api.py                  — REST API endpoints
├── views.py                — Flask-AppBuilder views and Pydantic schemas
├── capability_contract.py  — Rules, constants, contract evaluation
├── app.py                  — Flask-AppBuilder application factory
├── database/
│   └── store.py            — PostgreSQL store layer
├── domain/
│   ├── events.py           — Domain event definitions
│   ├── rules.py            — Business rule implementations
│   └── adapters.py         — External system adapters
├── alembic/                — Database migration scripts
├── tests/
│   ├── test_service.py     — Service unit tests
│   └── test_contract.py    — Contract compliance tests
├── docs/
│   └── user_guide.md       — This file
├── WORLD_CLASS_IMPROVEMENTS.md  — Roadmap for 15 system improvements
└── README.md               — Quick reference
```

---

## Interoperability

Reference in `.apg` composition files:

```apg
use government_eme;
```

Composes with:
- `government_law` — security incidents activate law enforcement dockets
- `government_bud` — emergency expenditures create budget commitments
- `government_csr` — citizen alerts and relief applications
- `intel` — threat intelligence feeds incident severity assessment
- `government_cas` — public emergency reports become cases
