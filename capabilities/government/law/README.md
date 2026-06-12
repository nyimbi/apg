# Law Enforcement and Justice

## Overview
Incident reporting with OB number generation, case docket management, evidence chain of custody, court scheduling, and prosecution tracking from arrest to conviction. Enforces strict chain-of-custody rules and requires DPP reference numbers before prosecution can commence.

## Capability ID
`government_law`

## Provides
- incident_reporting_workflow: OB number generation and incident capture
- docket_management_workflow: Case docket lifecycle from opening to closure
- evidence_chain_of_custody_workflow: Tamper-evident evidence tracking
- court_scheduling_workflow: Court calendar and hearing management
- prosecution_tracking_workflow: DPP referral and prosecution status
- law_enforcement_review_workflow: Governance and oversight reviews
- law_enforcement_agent_workflow: Automated docket and evidence management agents
- ob_number_generation_workflow: Sequential OB number allocation
- witness_management_workflow: Witness statement collection and management
- inter_agency_referral_workflow: Cross-agency case referrals

## Requires
| Capability | Reason |
|---|---|
| auth | Officer and detective RBAC |
| audl | Immutable law enforcement audit trail |
| mten | Tenant-scoped docket isolation |
| conf | Jurisdiction and court configuration |
| ntfy | Case status notifications to stakeholders |
| wflo | Docket state machine and approval flows |
| geos | Crime scene geolocation mapping |
| schd | Court hearing scheduling |
| moni | Docket SLA and overdue monitoring |
| mqeb | Event streaming via bytewax |

## Configuration
| Key | Description |
|---|---|
| governance.chain_of_custody_breach_denied | Any custody gap triggers denial |
| governance.evidence_tampering_denied | Evidence integrity always enforced |
| governance.prosecution_without_dpp_reference_denied | DPP reference mandatory for prosecution |
| evidence.chain_of_custody_enforced | Every evidence movement is logged |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /government-law/incidents | GET/POST | Incident reporting | government_law:incidents |
| /government-law/dockets | GET/POST | Docket management | government_law:dockets |
| /government-law/evidence | GET/POST | Evidence logging | government_law:evidence |
| /government-law/custody | GET/POST | Custody chain ledger | government_law:custody |
| /government-law/court-scheduling | GET/POST | Court hearings | government_law:court |
| /government-law/prosecution | GET/POST | Prosecution tracking | government_law:prosecution |
| /government-law/map | GET | Crime map view | government_law:view |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| ob_number_required | ob_number_present=False | deny |
| chain_of_custody_breach_denied | chain_intact=False | deny |
| prosecution_dpp_reference_required | dpp_reference_present=False | deny |
| evidence_reference_required | evidence_reference_present=False | deny |
| investigating_officer_required | investigating_officer_present=False | deny |

## Data Models
- IncidentReport: id, tenant_id, incident_type, ob_number, reporting_officer_id, location_reference
- CaseDocket: id, tenant_id, incident_id, investigating_officer_id, status, docket_number
- EvidenceItem: id, docket_id, evidence_type, custodian_id, evidence_reference, current_location
- CustodyAction: id, evidence_id, custody_action, actor_id, from_location, to_location
- CourtHearing: id, docket_id, court_type, hearing_type, hearing_date, presiding_judge
- ProsecutionRecord: id, docket_id, dpp_reference, prosecution_status, charges
- LawEnforcementReview, LawEnforcementAgent

## Streaming Events
- incident_reported, docket_opened, docket_status_changed, evidence_logged
- evidence_custody_action_recorded, court_hearing_scheduled, prosecution_status_updated, conviction_recorded

## Edge Cases Handled
- Evidence custody gap (missing custodian in chain) — any action on that evidence is denied
- DPP referral attempted without obtaining DPP reference number — denied
- Court hearing rescheduled after conviction — invalid state transition blocked
- Cross-jurisdiction docket access restricted by tenancy boundary
- Forensics lab transfer automatically updates `current_location` on evidence item

## Composability Notes
Composes with `government_eme` (security incident response creates law enforcement dockets), `government_cas` (investigation outcomes feed case resolution), `government_con` (fraud procurement cases trigger docket opening), `government_bud` (embezzlement cases reference budget records), and `intel` (crime pattern analysis for resource allocation intelligence).

---

## World-Class Enhancements (v2.0)

- **I1.** Law Enforcement Capability — World-Class Improvements
- **I2.** Async Service Layer
- **I3.** PostgreSQL-Backed Persistence
- **I4.** Cryptographic Evidence Integrity
- **I5.** Immutable Append-Only Audit Trail via Event Sourcing
- **I6.** Structured Domain Events with CloudEvents Envelope
- **I7.** Chain-of-Custody Graph — Directed Acyclic Graph Model
- **I8.** CIMS Integration Adapter
- **I9.** Automated Docket SLA Monitoring
- **I10.** Geospatial Crime Hotspot Analysis
- **I11.** Digital Evidence Hash Verification on Transfer
- **I12.** Warrant Lifecycle State Machine
- **I13.** Victim / Complainant Case Portal (Read-Only Scoped Token)
- **I14.** Bulk Evidence Import via Structured Manifest
- **I15.** ML-Assisted Incident Classification

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
