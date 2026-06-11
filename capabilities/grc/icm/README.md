# Incident & Crisis Management (grc_icm)

**Capability ID**: `grc_icm` | **Domain**: `grc` | **Version**: 1.1.0  
© 2025 Datacraft — Author: Nyimbi Odero

## Overview

Full-lifecycle incident response, crisis communication, business continuity
management (BCMS), compliance testing, and regulatory notification for the APG
platform. Standalone-deployable; composes with other APG capabilities via the
standard contract interface.

## Provides

| Service | Description |
|---------|-------------|
| `incident_lifecycle_management` | Report → triage → investigate → close |
| `case_management_workflow` | Case and evidence management |
| `incident_evidence_workflow` | Chain-of-custody evidence collection |
| `regulatory_notification_workflow` | GDPR/PCI-DSS/CBK regulatory dispatch |
| `post_incident_review_workflow` | Structured PIR for high/critical incidents |

## Requires

| Capability | Purpose |
|------------|---------|
| `auth` | Identity and permission checks |
| `audl` | Audit event logging |
| `mten` | Multi-tenancy context |
| `conf` | Tenant-scoped configuration |
| `ntfy` | Notification delivery |

## Installation

```bash
pip install apg-grc-icm
```

## Standalone Usage

```python
from apg_grc_icm import IncidentComplianceService

svc = IncidentComplianceService()
inc = await svc.report_incident(
    entity_id="ENT-1",
    incident_type="security_breach",
    description="Phishing attack",
    severity="high",
    affected_systems=["email"],
    reported_by="alice@datacraft.co.ke",
)
```

## Running the Standalone Server

```bash
# In-memory store (development)
apg-grc-icm --port 8080

# With PostgreSQL persistence
apg-grc-icm --db-url postgresql+asyncpg://user:pass@localhost/icm --port 8080
```

## Service Methods

### Incident Lifecycle
`report_incident`, `incident_triage`, `incident_investigation`, `root_cause_analysis`,
`corrective_action`, `corrective_action_update`, `corrective_action_verify`,
`close_incident`, `incident_reopen`, `incident_escalate`, `incident_categorise`

### Investigation & Evidence
`investigation_assign`, `verify_evidence_chain`, `find_similar_incidents`

### Playbooks
`activate_playbook`, `advance_playbook_task`

### Business Continuity
`business_continuity_activation`, `business_impact_assessment`, `bcp_activate`

### Crisis Communication
`create_war_room`, `war_room_post`, `close_war_room`, `communication_log`, `notification_send_icm`

### Regulatory & Compliance
`regulatory_notification`, `third_party_incident_notify`, `vendor_acknowledgement_record`,
`compliance_test`, `compliance_deficiency`, `remediation_plan`, `compliance_evidence`,
`compliance_score`, `compliance_calendar`

### Reporting & Analytics
`incident_analytics`, `incident_kpi_summary`, `compliance_dashboard`,
`regulatory_reporting_icm`, `get_sla_status`, `generate_executive_briefing`

### Post-Incident
`post_incident_review`, `lessons_learned_capture`, `lessons_learned_library`,
`preventive_action_plan`, `insurance_claim_trigger`, `root_cause_confirm`

### ML Enhancements
`ml_incident_severity` — Ollama-powered severity classification (optional)

## API Routes

| Name | Path | Permission |
|------|------|------------|
| dashboard | `/grc-icm/dashboard` | `grc_icm:view` |
| incidents | `/grc-icm/incidents` | `grc_icm:manage_incidents` |
| incident_detail | `/grc-icm/incidents/:id` | `grc_icm:view` |
| cases | `/grc-icm/cases` | `grc_icm:manage_cases` |
| case_detail | `/grc-icm/cases/:id` | `grc_icm:view` |
| evidence | `/grc-icm/evidence` | `grc_icm:manage_evidence` |
| notifications | `/grc-icm/notifications` | `grc_icm:view` |
| timeline | `/grc-icm/timeline` | `grc_icm:view` |

## HTTP Endpoints

```
GET  /health           Liveness probe
GET  /contract         Full capability contract JSON
POST /evaluate         Evaluate governance rules
GET  /api/v1/...       Domain-specific REST API
```

## Composability

```python
from capabilities.capability_contract_registry import load_contract_registry
registry = load_contract_registry()
contract = registry["grc_icm"].contract
```

APG DSL: `use grc_icm;`

## Development

```bash
pytest tests/ -q
uv run pyright
python -m build --wheel .
```

## License

Proprietary — © 2025 Datacraft  
Author: Nyimbi Odero <nyimbi@gmail.com>
