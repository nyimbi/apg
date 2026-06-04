# Electoral and Civil Registration

## Overview
Voter registration with biometric deduplication, polling station management, election results collation, and civil registry for births, deaths, marriages, and other vital events. Enforces integrity rules that prevent duplicate voter registration, underage registration, and result manipulation.

## Capability ID
`government_ele`

## Provides
- voter_registration_workflow: Voter registration with biometric capture
- biometric_deduplication_workflow: Multi-method deduplication (fingerprint, iris, facial)
- polling_station_management_workflow: Station assignment and officer allocation
- election_management_workflow: Election creation and constituency management
- results_collation_workflow: Polling station results aggregation and announcement
- civil_registration_workflow: Birth, death, marriage, and adoption registration
- electoral_verification_workflow: Voter identity verification at polls
- electoral_review_workflow: Governance review of electoral processes
- electoral_agent_workflow: Automated registration and deduplication agents
- civil_registry_amendment_workflow: Amendment and late registration processing

## Requires
| Capability | Reason |
|---|---|
| auth | Electoral officer authentication and RBAC |
| audl | Tamper-evident audit of all electoral actions |
| mten | Tenant-scoped electoral data isolation |
| conf | Biometric threshold and deduplication configuration |
| ntfy | Voter notification of registration status |
| geos | Constituency boundary management |
| comp | Electoral Act compliance checks |
| moni | Real-time registration and results monitoring |
| mqeb | Event streaming via bytewax |

## Configuration
| Key | Description |
|---|---|
| deduplication.duplicate_detection_threshold | 0.95 match score triggers duplicate flag |
| deduplication.primary_method | biometric_fingerprint |
| governance.duplicate_voter_denied | Block duplicate registration |
| governance.underage_voter_denied | Minimum voting age enforcement |
| governance.result_manipulation_denied | Cryptographic result integrity |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /government-ele/registrations | GET/POST | Voter registration | government_ele:register |
| /government-ele/deduplication | GET/POST | Deduplication console | government_ele:deduplicate |
| /government-ele/polling-stations | GET/POST | Station management | government_ele:stations |
| /government-ele/results | GET/POST | Results collation | government_ele:results |
| /government-ele/civil-registry | GET/POST | Civil registry | government_ele:civil |
| /government-ele/boundaries | GET | Constituency map | government_ele:boundaries |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| duplicate_voter_denied | duplicate_detected=True | deny |
| underage_voter_denied | of_voting_age=False | deny |
| voter_biometric_required | biometric_present=False | deny |
| result_manipulation_denied | manipulation_detected=True | deny |
| cross_constituency_result_denied | cross_constituency=True | deny |

## Data Models
- VoterRegistration: id, tenant_id, national_id, biometric_reference, constituency, polling_station_id
- DeduplicationRecord: id, registration_id, method, match_score, duplicate_detected
- PollingStation: id, tenant_id, station_type, constituency, capacity, presiding_officer_id
- Election: id, tenant_id, election_type, polling_date, constituency, status
- ElectionResult: id, election_id, polling_station_id, candidate_id, votes_cast, status
- CivilRegistryEvent: id, tenant_id, registration_type, subject_id, registrar_id, event_date
- ElectoralVerification, ElectoralReview, ElectoralAgent

## Streaming Events
- voter_registered, duplicate_detected, duplicate_resolved, polling_station_assigned
- election_results_collated, civil_event_registered, voter_verified, result_announced

## Edge Cases Handled
- Biometric match score below threshold triggers manual review, not automatic denial
- Diaspora polling stations have different documentation requirements
- Late civil registration allowed with registrar justification
- Cross-constituency result submission always denied even for admin users
- Civil registry amendments require original registrar and new evidence

## Composability Notes
Composes with `government_csr` (voter card applications processed through portal), `government_cas` (electoral complaints become cases), `government_law` (electoral offences create police dockets), and `intel` (voter pattern analytics for constituency planning).
