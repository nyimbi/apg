# Electoral and Civil Registration

## Overview
Voter registration with biometric deduplication, polling station management, election results collation,
and civil registry for births, deaths, marriages, and other vital events.  Enforces integrity rules
that prevent duplicate voter registration, underage registration, and result manipulation.

Version 2.0 adds: zero-knowledge voter verification, NATS-backed real-time result streaming,
AI-assisted multi-modal deduplication, offline-first mobile station sync, statistical anomaly
detection, ranked-choice tabulation, Merkle-tree tamper-evident voter roll, automated compliance
auditing, and candidate eligibility cross-referencing.

## Capability ID
`government_ele`

## Provides
- voter_registration_workflow: Voter registration with biometric capture
- biometric_deduplication_workflow: Multi-method deduplication (fingerprint, iris, facial) + AI scoring
- polling_station_management_workflow: Station assignment, officer allocation, offline queue
- election_management_workflow: Election creation, constituency management, ranked-choice tabulation
- results_collation_workflow: Polling station results aggregation and announcement
- results_streaming_workflow: Real-time NATS JetStream result publishing
- civil_registration_workflow: Birth, death, marriage, and adoption registration
- electoral_verification_workflow: Voter identity verification at polls + ZK credential proofs
- electoral_review_workflow: Governance review of electoral processes
- electoral_agent_workflow: Automated registration and deduplication agents
- civil_registry_amendment_workflow: Amendment and late registration processing
- compliance_audit_workflow: Automated Electoral Act compliance reporting

## Requires
| Capability | Reason |
|---|---|
| auth | Electoral officer authentication and RBAC |
| audl | Tamper-evident audit of all electoral actions |
| mten | Tenant-scoped electoral data isolation |
| conf | Biometric threshold and deduplication configuration |
| ntfy | Voter notification of registration status (NATS-driven) |
| geos | Constituency boundary management |
| comp | Electoral Act compliance checks |
| moni | Real-time registration and results monitoring |
| mqeb | Event streaming via bytewax + NATS |

## Configuration
| Key | Description |
|---|---|
| deduplication.duplicate_detection_threshold | 0.95 match score triggers duplicate flag |
| deduplication.primary_method | biometric_fingerprint |
| deduplication.ai_model | ollama/llava-biometric (default) |
| governance.duplicate_voter_denied | Block duplicate registration |
| governance.underage_voter_denied | Minimum voting age enforcement |
| governance.result_manipulation_denied | Cryptographic result integrity |
| merkle.publish_on_build | Publish root hash to NATS on tree build |
| compliance.legal_framework | electoral_act (default) |
| offline.lamport_sync | true — preserve causal operation ordering |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /government-ele/registrations | GET/POST | Voter registration | government_ele:register |
| /government-ele/deduplication | GET/POST | Deduplication console | government_ele:deduplicate |
| /government-ele/polling-stations | GET/POST | Station management | government_ele:stations |
| /government-ele/results | GET/POST | Results collation | government_ele:results |
| /government-ele/civil-registry | GET/POST | Civil registry | government_ele:civil |
| /government-ele/boundaries | GET | Constituency map | government_ele:boundaries |
| /government-ele/merkle | GET | Voter roll Merkle root | government_ele:audit |
| /government-ele/compliance | GET | Compliance audit report | government_ele:audit |
| /government-ele/anomalies | GET | Statistical anomaly report | government_ele:audit |
| /government-ele/ranked-choice | POST | Ranked-choice tabulation | government_ele:results |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| duplicate_voter_denied | duplicate_detected=True | deny |
| underage_voter_denied | of_voting_age=False | deny |
| voter_biometric_required | biometric_present=False | deny |
| result_manipulation_denied | manipulation_detected=True | deny |
| cross_constituency_result_denied | cross_constituency=True | deny |
| candidate_hard_fail_blocks_registration | eligible=False | deny |
| anomaly_quarantine | anomalies_detected>0 | quarantine |

## Key Service Methods

### Existing (synchronous)
- `register_voter()` — Biometric registration with dedup enforcement
- `voter_registration()` — Simplified citizen-facing registration
- `biometric_capture()` — Capture + quality scoring
- `polling_station_setup()` — Station configuration
- `voter_list_verification()` — Constituency voter roll check
- `ballot_management()` — Ballot type and serial range definition
- `vote_counting()` — Station tally recording
- `result_collation()` — Constituency-level aggregation
- `result_transmission()` — Secure tally transmission
- `election_analytics()` — In-progress/completed election statistics
- `audit_trail()` — Complete tamper-evident audit log
- `observer_accredit()` — Observer accreditation and badging
- `turnout_calculate()` — Voter turnout computation
- `anomaly_flag()` — Manual anomaly flagging
- `voter_purge()` — Deceased/ineligible voter removal

### New Async Methods (v2.0)
- `verify_zk_credential()` — ZK proof voter eligibility check, zero PII exposure
- `stream_result_updates()` — NATS JetStream signed result publishing
- `ai_deduplication_score()` — Ollama-backed multi-modal biometric dedup scoring
- `queue_offline_operation()` — Offline-first operation queuing with Lamport clock
- `sync_offline_queue()` — Drain offline queue over NATS in causal order
- `detect_statistical_anomalies()` — Benford's Law + Z-score SPC anomaly detection
- `notify_voter_status_change()` — Multi-channel voter notification via NATS→ntfy
- `build_voter_roll_merkle_tree()` — SHA-256 Merkle tree for tamper-evident voter roll
- `tabulate_ranked_choice()` — Instant-runoff ranked-choice multi-round tabulation
- `validate_candidate_eligibility()` — Cross-capability eligibility verification
- `run_compliance_audit()` — Automated Electoral Act compliance reporting

## Data Models
- VoterRegistration: id, tenant_id, national_id, biometric_reference, constituency, polling_station_id
- DeduplicationRecord: id, registration_id, method, match_score, duplicate_detected
- PollingStation: id, tenant_id, station_type, constituency, capacity, presiding_officer_id
- Election: id, tenant_id, election_type, polling_date, constituency, status
- ElectionResult: id, election_id, polling_station_id, candidate_id, votes_cast, status
- CivilRegistryEvent: id, tenant_id, registration_type, subject_id, registrar_id, event_date
- ElectoralVerification, ElectoralReview, ElectoralAgent

## Streaming Events (NATS subjects)
| Subject | Trigger |
|---|---|
| apg.government.ele.results.{constituency_id} | result_collation, stream_result_updates |
| apg.government.ele.notifications.{voter_id} | notify_voter_status_change |
| apg.government.ele.merkle.roots | build_voter_roll_merkle_tree |
| apg.government.ele.compliance.{election_id} | run_compliance_audit |
| apg.government.ele.offline_sync | sync_offline_queue |
| apg.government.ele.lifecycle | voter_registered, civil_event_registered, etc. |

## Edge Cases Handled
- Biometric match score below threshold triggers manual review, not automatic denial
- Diaspora polling stations have different documentation requirements
- Late civil registration allowed with registrar justification
- Cross-constituency result submission always denied even for admin users
- Civil registry amendments require original registrar and new evidence
- Offline polling stations queue operations with Lamport timestamps for causal sync
- Ranked-choice tabulation falls back to plurality if no majority is reachable
- ZK proof verification never persists or returns voter PII

## Composability Notes
Composes with `government_csr` (voter card applications), `government_cas` (electoral complaints
become cases), `government_law` (electoral offences create police dockets), and `intel` (voter
pattern analytics for constituency planning).

Candidate eligibility checks compose with `government_law` (criminal disqualifications) and the
`civil_events` registry (age, citizenship).

## Further Reading
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 detailed improvement plans with competitor analysis
- `service.py` — Business logic implementation
- `models.py` — Data models
- `docs/user_guide.md` — Full operator and developer guide
- `cap_spec.md` — Capability specification
- `SPECIFICATION.md` — Detailed functional specification
