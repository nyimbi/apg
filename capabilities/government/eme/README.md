# Emergency Management

## Overview
Incident command, resource mobilisation, multi-agency coordination, EOC management, situation reporting, evacuation, relief distribution, casualty tracking, damage assessment, and after-action reviews. Implements the Incident Command System (ICS) and NIMS frameworks with AI-assisted decision support, CAP-compliant public alerting, and NATS-backed event choreography.

## Capability ID
`government_eme`

## Provides
- `incident_command_workflow`: Declare, escalate, and close emergency incidents
- `resource_mobilisation_workflow`: Deploy, track, and predict resource gaps
- `multi_agency_coordination_workflow`: Activate and coordinate responding agencies
- `eoc_management_workflow`: Emergency Operations Centre activation and management
- `situation_reporting_workflow`: Structured situation reports (SITREPs), AI-drafted
- `after_action_review_workflow`: Post-incident lessons-learned with SAIR improvement tracking
- `emergency_review_workflow`: Governance review of emergency response
- `emergency_agent_workflow`: Automated coordination and reporting agents
- `incident_phase_transition_workflow`: Manage incident lifecycle phases
- `resource_demobilisation_workflow`: Stand-down and return of resources
- `cap_alert_workflow`: CAP v1.2-compliant public alert broadcasting
- `mutual_aid_workflow`: EMAC-format inter-jurisdictional mutual aid requests
- `cross_capability_choreography_workflow`: CloudEvents fan-out to peer APG capabilities

## Requires
| Capability | Reason |
|---|---|
| auth | Incident commander and officer RBAC |
| audl | Chain-of-command audit trail |
| mten | Tenant-scoped incident data |
| conf | Incident thresholds and escalation rules |
| ntfy | Multi-agency activation notifications |
| geos | Incident geolocation and mapping |
| moni | Real-time resource and incident monitoring |
| schd | Resource scheduling and deployment planning |
| mqeb | Event streaming via bytewax+NATS |

## Configuration
| Key | Description |
|---|---|
| governance.unauthorised_eoc_activation_denied | Only authorised officials can activate EOC |
| governance.after_action_mandatory_post_incident | AAR required before incident closure |
| governance.resource_over_allocation_denied | Prevent double-booking of resources |
| incidents.supported_phases | detection, notification, activation, response, recovery, stand_down, after_action |
| eme.ollama_base_url | Ollama endpoint for ML severity scoring and SITREP generation |
| eme.nats_url | NATS JetStream URL for event streaming and alert broadcasting |
| eme.mutual_aid_jurisdictions | JSON map of jurisdiction codes to webhook endpoints |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /government-eme/incidents | GET/POST | Incident command console | government_eme:incidents |
| /government-eme/resources | GET/POST | Resource mobilisation | government_eme:resources |
| /government-eme/agencies | GET/POST | Agency coordination | government_eme:agencies |
| /government-eme/eoc | GET/POST | EOC management | government_eme:eoc |
| /government-eme/situation-reports | GET/POST | SITREPs | government_eme:reports |
| /government-eme/after-action | GET/POST | After-action reviews | government_eme:aar |
| /government-eme/alerts | POST | CAP alert broadcasting | government_eme:alerts |
| /government-eme/mutual-aid | POST | Mutual aid requests | government_eme:mutual_aid |
| /government-eme/shelters | GET/POST | Shelter management | government_eme:shelters |
| /government-eme/icp-picture | GET | Live ICP GeoJSON picture | government_eme:view |
| /government-eme/map | GET | Incident situation map | government_eme:view |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| incident_commander_required | commander_present=False | deny |
| unauthorised_eoc_activation_denied | authorised=False | deny |
| resource_over_allocation_denied | over_allocated=True | deny |
| aar_lessons_required | lessons_present=False | deny |
| aar_reviewer_required | reviewer_present=False | deny |

## Data Models
- `EmergencyIncident`: id, tenant_id, incident_type, severity, phase, location_reference, commander_id
- `ResourceMobilisation`: id, incident_id, resource_type, quantity, responsible_agency, status
- `AgencyActivation`: id, incident_id, agency_type, agency_name, contact_reference, role
- `EocRecord`: id, incident_id, eoc_status, command_structure, activation_authority
- `SituationReport`: id, incident_id, period, author_id, summary
- `AfterActionReview`: id, incident_id, reviewer_id, lessons_learned, recommendations
- `EmergencyReview`, `EmergencyAgent`

## Sync Service Methods
| Method | Description |
|---|---|
| `declare_incident()` | Full-parameter incident declaration |
| `declare_emergency()` | Shorthand emergency declaration with auto-EOC |
| `activate_eoc()` | Emergency Operations Centre activation |
| `resource_mobilisation()` | Batch resource mobilisation |
| `multi_agency_coordination()` | Batch agency activation |
| `situation_report()` | SITREP generation from incident state |
| `evacuation_management()` | Zone evacuation management |
| `relief_distribution()` | Relief item distribution to locations |
| `casualty_tracking()` | Casualty summary by status |
| `after_action_review()` | Post-incident AAR |
| `public_alert()` | Basic public alert issuance |
| `damage_assess()` | Damage category recording |
| `mutual_aid_request()` | Mutual aid request stub |
| `incident_close()` | Formal incident closure |
| `emergency_analytics()` | Period-based analytics |

## Async Service Methods (New)
| Method | Description |
|---|---|
| `async_broadcast_cap_alert()` | CAP v1.2-compliant multi-channel alert broadcast via NATS |
| `async_predict_resource_gaps()` | ML-based resource exhaustion prediction with NATS alerts |
| `async_generate_sitrep_narrative()` | Ollama-powered ICS-209 narrative draft generation |
| `async_update_resource_position()` | AVL GPS position update returning GeoJSON Feature |
| `async_match_volunteers()` | Skill-matching engine for volunteer assignment |
| `async_update_shelter_occupancy()` | Check-in/check-out with capacity warning alerts |
| `async_replay_incident_timeline()` | JetStream event replay for incident reconstruction |
| `async_publish_cross_capability_events()` | CloudEvents fan-out to government_law, intel, etc. |
| `async_submit_mutual_aid_request()` | EMAC-format mutual aid to neighbouring jurisdiction |
| `async_escalate_incident()` | Severity escalation with downstream notifications |
| `async_render_icp_picture()` | GeoJSON ICP common picture for MapLibre rendering |
| `async_create_improvement_action()` | SAIR improvement action tracking from AAR findings |

## Streaming Events (NATS Subjects)
| Subject | Event |
|---|---|
| `eme.events.{tenant_id}.{incident_id}` | All incident lifecycle events (JetStream durable) |
| `eme.broadcast.{channel}` | CAP alert fan-out (sms, push, eas, ussd) |
| `eme.alerts.escalation.{incident_id}` | Severity escalation notifications |
| `eme.alerts.resource_gap.{incident_id}` | Predicted resource shortage warnings |
| `eme.alerts.shelter_capacity.{shelter_id}` | Shelter over-capacity warnings |
| `eme.avl.{resource_id}` | Inbound GPS telemetry from field units |
| `eme.mutual_aid.outbound.{jurisdiction}` | EMAC mutual aid requests |
| `eme.mutual_aid.status.{request_id}` | Mutual aid response callbacks |
| `eme.icp.{incident_id}.picture` | Live ICP GeoJSON picture (60s interval) |
| `apg.{capability}.events.eme_triggered` | Cross-capability choreography events |

## Edge Cases Handled
- Unauthorised EOC activation denied even for senior officers without activation authority
- After-action review enforced before incident can be fully closed
- Resource over-allocation across multiple incidents prevented
- Multi-agency coordination maintains separate evidence chains per agency
- Catastrophic incidents trigger automatic escalation to national EOC
- Severity escalation validates new level is strictly higher than current
- CAP alerts fall back to template broadcast when NATS is unavailable
- SITREP generation falls back to structured template when Ollama is unreachable

## Composability Notes
Composes with `government_law` (security incidents activate law enforcement dockets), `government_cas` (public emergency reports become cases), `government_bud` (emergency resource costs create budget commitments), `government_csr` (citizen alerts and relief applications), and `intel` (threat intelligence feeds incident severity assessment). Cross-capability events are published automatically via `async_publish_cross_capability_events()`.

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Real-Time Predictive Incident Escalation** [AI/ML Decision Support]
- **I2. CAP-Compliant Public Alert Broadcasting** [Public Warning Standards]
- **I3. GIS-Integrated Damage Assessment with Satellite Change Detection** [Geospatial Intelligence]
- **I4. NATS-Backed Event Sourcing for Full Incident Timeline** [Data Architecture / Auditability]
- **I5. Unified Resource Tracking with QR/RFID Position Updates** [Resource Management]
- **I6. Inter-Jurisdictional Mutual Aid Workflow Automation** [Interoperability]
- **I7. Predictive Resource Gap Analysis** [Logistics Intelligence]
- **I8. AI-Assisted SITREP Generation** [Reporting Automation]
- **I9. Volunteer Skill-Matching Engine** [Human Capital Management]
- **I10. Automated Shelter Capacity Management** [Mass Care Logistics]
- **I11. Multi-Modal Communication Resilience** [Communications / Redundancy]
- **I12. Incident Command Post Digital Twin** [Situational Awareness]
- **I13. Compliance-Driven After-Action Workflow** [Governance / Continuous Improvement]
- **I14. Real-Time Casualty De-duplication and Family Reunification** [Life Safety / Data Quality]
- **I15. NATS-Driven Cross-Capability Event Choreography** [Composability / Integration]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
