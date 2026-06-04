# Emergency Management

## Overview
Incident command, resource mobilisation, multi-agency coordination, EOC management, situation reporting, and after-action reviews. Implements the Incident Command System (ICS) framework with mandatory after-action reviews and strict EOC activation authority controls.

## Capability ID
`government_eme`

## Provides
- incident_command_workflow: Declare and manage emergency incidents
- resource_mobilisation_workflow: Deploy and track resources to incidents
- multi_agency_coordination_workflow: Activate and coordinate responding agencies
- eoc_management_workflow: Emergency Operations Centre activation and management
- situation_reporting_workflow: Structured situation reports (SITREPs)
- after_action_review_workflow: Post-incident lessons-learned documentation
- emergency_review_workflow: Governance review of emergency response
- emergency_agent_workflow: Automated coordination and reporting agents
- incident_phase_transition_workflow: Manage incident lifecycle phases
- resource_demobilisation_workflow: Stand-down and return of resources

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
| mqeb | Event streaming via bytewax |

## Configuration
| Key | Description |
|---|---|
| governance.unauthorised_eoc_activation_denied | Only authorised officials can activate EOC |
| governance.after_action_mandatory_post_incident | AAR required before incident closure |
| governance.resource_over_allocation_denied | Prevent double-booking of resources |
| incidents.supported_phases | detection, notification, activation, response, recovery, stand_down, after_action |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /government-eme/incidents | GET/POST | Incident command console | government_eme:incidents |
| /government-eme/resources | GET/POST | Resource mobilisation | government_eme:resources |
| /government-eme/agencies | GET/POST | Agency coordination | government_eme:agencies |
| /government-eme/eoc | GET/POST | EOC management | government_eme:eoc |
| /government-eme/situation-reports | GET/POST | SITREPs | government_eme:reports |
| /government-eme/after-action | GET/POST | After-action reviews | government_eme:aar |
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
- EmergencyIncident: id, tenant_id, incident_type, severity, phase, location_reference, commander_id
- ResourceMobilisation: id, incident_id, resource_type, quantity, responsible_agency, status
- AgencyActivation: id, incident_id, agency_type, agency_name, contact_reference, role
- EocRecord: id, incident_id, eoc_status, command_structure, activation_authority
- SituationReport: id, incident_id, period, author_id, summary
- AfterActionReview: id, incident_id, reviewer_id, lessons_learned, recommendations
- EmergencyReview, EmergencyAgent

## Streaming Events
- incident_declared, incident_phase_transitioned, resource_mobilised, resource_demobilised
- agency_activated, eoc_activated, situation_report_filed, incident_stood_down, after_action_review_completed

## Edge Cases Handled
- Unauthorised EOC activation denied even for senior officers without activation authority
- After-action review enforced before incident can be fully closed
- Resource over-allocation across multiple incidents prevented
- Multi-agency coordination maintains separate evidence chains per agency
- Catastrophic incidents trigger automatic escalation to national EOC

## Composability Notes
Composes with `government_law` (security incidents activate law enforcement dockets), `government_cas` (public emergency reports become cases), `government_bud` (emergency resource costs create budget commitments), `government_csr` (citizen alerts and relief applications), and `intel` (threat intelligence feeds incident severity assessment).
