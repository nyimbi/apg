# Construction Management (realestate_con)

## Overview
Full construction project and contract lifecycle management: project management, contractor management, defect tracking, snagging, and handover workflows. Covers contract drafting through practical completion, payment certificate management, drawing register, risk register, extension of time claims, snagging workflows, and a searchable clause library.

## Capability ID
`realestate_con`

## Provides
- `contract_lifecycle_management`: Draft through execution, suspension, termination, PC certificate
- `contractor_registry_management`: Graded contractor panel (preferred to blacklisted) with performance scorecard
- `milestone_tracking_workflow`: Typed milestones with due-date alerts and evidence capture
- `variation_order_management`: Scope/price/timeline variations with board-threshold gate
- `dispute_resolution_workflow`: Typed disputes with legal review and resolution tracking
- `contract_clause_library`: Searchable standard and custom clauses by type and tag
- `retention_management`: Percentage/fixed retention with defect-liability-clearance gate
- `contract_expiry_alerts`: Rolling expiry pipeline with configurable advance warning
- `digital_signature_workflow`: Multi-party signature tracking with method support
- `contract_performance_reporting`: Value, milestone, variation, and dispute KPIs
- `snagging_workflow`: Create, track, and resolve defect items with severity-based SLAs
- `payment_certificate_workflow`: Interim payment certificates with retention and advance deductions
- `risk_register`: Project risk register with probability/impact scoring and risk score
- `drawing_register`: Revision-controlled drawing register with discipline filtering
- `eot_management`: Extension of Time claim submission, eligibility assessment, and milestone extension
- `practical_completion`: PC certificate issuance with snag validation and DLP tracking

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | Signing authority and legal review |
| `audl` | Immutable audit for all contract events |
| `mten` | Tenant isolation |
| `conf` | Board-approval threshold configuration |
| `ntfy` | Expiry alerts, milestone overdue notifications, snag SLA alerts |
| `wflo` | Board approval for large variations |
| `nlpc` | Clause library semantic search |
| `comp` | Construction law compliance (JBCC, NEC4, FIDIC) |
| `mqeb` | Publish construction lifecycle events via NATS |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| `variations.board_approval_threshold` | 500,000 | KES amount requiring board approval |
| `retention.default_percentage` | 5.0 | Default retention % of contract value |
| `contractors.grading_review_months` | 12 | Months between contractor grade reviews |
| `snags.sla_critical_days` | 2 | Resolution SLA days for critical snags |
| `snags.sla_major_days` | 7 | Resolution SLA days for major snags |
| `snags.sla_minor_days` | 14 | Resolution SLA days for minor snags |
| `eot.default_notice_period_days` | 28 | NEC4 compensation event notice window |
| `dlp.default_months` | 12 | Default defects liability period (months) |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/con/contracts` | GET/POST | List/create contracts | `contracts` |
| `/realestate/con/contracts/<id>/execute` | POST | Execute contract | `contracts` |
| `/realestate/con/contracts/<id>/terminate` | POST | Terminate contract | `contracts` |
| `/realestate/con/contracts/<id>/sign/<party>` | POST | Record signature | `contracts` |
| `/realestate/con/contracts/<id>/close` | POST | Close contract (PC) | `contracts` |
| `/realestate/con/expiry` | GET | Expiry pipeline | `view` |
| `/realestate/con/contractors` | GET/POST | Contractor registry | `contractors` |
| `/realestate/con/contractors/<id>/grade` | POST | Grade contractor | `contractors` |
| `/realestate/con/milestones` | GET/POST | Milestones | `milestones` |
| `/realestate/con/variations` | GET/POST | Variation orders | `variations` |
| `/realestate/con/variations/<id>/approve` | POST | Approve variation | `variations` |
| `/realestate/con/disputes` | GET/POST | Disputes | `disputes` |
| `/realestate/con/retention` | POST | Create retention | `retention` |
| `/realestate/con/retention/<id>/release` | POST | Release retention | `retention` |
| `/realestate/con/clauses` | GET/POST | Clause library | `clauses` |
| `/realestate/con/snags` | GET/POST | Snag list | `snags` |
| `/realestate/con/snags/<id>/resolve` | POST | Resolve snag | `snags` |
| `/realestate/con/snags/summary` | GET | Snag summary by severity/trade | `snags` |
| `/realestate/con/payment-certs` | POST | Issue payment certificate | `finance` |
| `/realestate/con/risks` | GET/POST | Risk register | `risks` |
| `/realestate/con/drawings` | GET/POST | Drawing register | `drawings` |
| `/realestate/con/drawings/current` | GET | Current drawing set | `drawings` |
| `/realestate/con/eot` | GET/POST | EOT claims | `contracts` |
| `/realestate/con/eot/<id>/assess` | POST | Assess EOT claim | `contracts` |
| `/realestate/con/pc-cert` | POST | Issue PC certificate | `contracts` |

## Service Methods — Full Reference
| Method | Description |
|--------|-------------|
| `create_contract()` | Create a contract record from a `ContractCreate` payload |
| `get_contract()` | Fetch a contract by ID and tenant |
| `list_contracts()` | List contracts with optional type/status filters |
| `update_contract()` | Update mutable contract fields |
| `execute_contract()` | Execute contract after all signatures and legal review |
| `terminate_contract()` | Terminate with reason and notice period |
| `sign_contract_party()` | Record a party's signature |
| `get_expiry_pipeline()` | Contracts expiring within N days |
| `draft_contract()` | Draft contract from template |
| `contract_review()` | Record legal/commercial/technical review |
| `contract_obligation_tracking()` | Track contractual obligations |
| `contract_milestone()` | Record milestone achievement |
| `variation_order()` | Create and approve VO in one step |
| `contract_close()` | Close contract on practical completion |
| `contract_analytics()` | Portfolio analytics for a period |
| `default_notice()` | Serve formal default notice |
| `dispute_management()` | Initiate and track dispute with resolution pathway |
| `register_contractor()` | Register contractor in registry |
| `get_contractor()` | Fetch a contractor |
| `list_contractors()` | List contractors by grade |
| `grade_contractor()` | Update contractor grade |
| `create_milestone()` | Create a contract milestone |
| `complete_milestone()` | Mark milestone as completed |
| `get_overdue_milestones()` | Return overdue milestones |
| `list_milestones()` | List milestones by contract |
| `raise_variation()` | Raise a variation order |
| `approve_variation()` | Approve a variation order |
| `list_variations()` | List variation orders |
| `raise_dispute()` | Raise a contract dispute |
| `resolve_dispute()` | Resolve a contract dispute |
| `list_disputes()` | List disputes |
| `create_retention()` | Create retention record |
| `release_retention()` | Release retention after DLP clearance |
| `create_clause()` | Add clause to library |
| `search_clauses()` | Search clause library |
| `get_contract_summary()` | High-level portfolio summary |
| `create_snag_item()` | Create snagging/defect item with severity SLA |
| `resolve_snag_item()` | Mark snag as resolved with evidence |
| `get_snag_list()` | Filtered snag list |
| `get_snag_summary()` | Snag counts by status/severity/trade |
| `issue_payment_certificate()` | Issue IPC with retention and net certified |
| `register_risk()` | Add risk to project risk register |
| `get_risk_register()` | Risk register filtered by score/status |
| `register_drawing()` | Register drawing revision, auto-supersede previous |
| `get_current_drawing_set()` | Current revisions only, by discipline |
| `issue_practical_completion_certificate()` | Issue PC cert with snag validation |
| `submit_extension_of_time()` | Submit EOT claim with cause category |
| `assess_extension_of_time()` | Assess EOT, extend affected milestone dates |
| `export_records()` | Export records as JSON or CSV |
| `health_check()` | Service health check |
| `compliance_audit()` | Run compliance audit against standard |
| `bulk_update_records()` | Bulk update records |
| `get_kpis()` | Compute service KPIs |
| `search_records()` | Full-text search across records |
| `ml_construction_risk()` | AI-powered cost overrun / delay risk (Ollama) |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `contract_requires_parties` | < 2 parties | deny (Pydantic) |
| `execution_requires_all_signatures` | unsigned parties | deny |
| `execution_requires_legal_review` | not reviewed | deny |
| `blacklisted_contractor_engagement_denied` | grade = blacklisted | deny |
| `variation_above_threshold_requires_board` | > 500k, no board | deny |
| `retention_release_requires_defect_clearance` | not cleared | deny |
| `termination_requires_reason` | no reason | deny |
| `termination_requires_notice_period` | notice not satisfied | deny |
| `pc_cert_requires_snag_clearance` | open snags > threshold | ValueError |
| `pc_cert_requires_commissioning` | commissioning_complete=False | ValueError |
| `eot_contractor_risk_ineligible` | cause_category=contractor_risk | logged warning, eligible=False |
| `snag_sla_assigned_by_severity` | critical/major/minor/observation | auto due_date |

## Data Models
- `ContractCreate/Response` — full contract with typed parties, governing law, value
- `ContractParty` — party with role, signature method, and signed-at timestamp
- `ContractorCreate/Response` — graded contractor with insurance and performance score
- `MilestoneCreate/Response` — typed milestone with due date, amount, evidence list
- `VariationOrderCreate/Response` — variation with amount/timeline change and board status
- `DisputeCreate/Response` — typed dispute with legal review flag and resolution summary
- `RetentionCreate/Response` — retention by method with defect liability end date
- `ClauseCreate/Response` — clause with type, tags, and usage count

## Streaming Events (NATS)
Published to NATS JetStream; subjects follow `con.<entity>.<action>` pattern.

- `con.contract.created`, `con.contract.executed`, `con.contract.suspended`, `con.contract.terminated`
- `con.contract.pc_cert_issued`
- `con.contractor.registered`, `con.contractor.graded`
- `con.milestone.reached`, `con.milestone.overdue`
- `con.variation.raised`, `con.variation.approved`, `con.variation.rejected`
- `con.dispute.raised`, `con.dispute.resolved`
- `con.retention.released`
- `con.snag.created`, `con.snag.resolved`, `con.snag.sla_breached`
- `con.drawing.registered`, `con.drawing.superseded`
- `con.eot.submitted`, `con.eot.granted`, `con.eot.rejected`
- `con.notice.default_served`, `con.notice.response_overdue`
- `con.payment_cert.issued`

## Edge Cases Handled
- Contract with only one party rejected at Pydantic model level
- Execution blocked until all party signatures recorded AND legal review complete
- Variation against draft contract blocked (must be active)
- Board approval tracked separately from standard approval on variation
- Retention release requires both defect clearance AND approval
- Blacklisted contractor engagement denied even if contractor is in registry
- PC certificate blocked if open snag count exceeds allowed threshold
- PC certificate blocked if commissioning_complete is False
- Drawing register auto-supersedes previous revision on same drawing number
- EOT claims with contractor-risk cause category flagged as ineligible but still stored
- Snag SLA due dates automatically assigned by severity on creation

## Composability Notes
- Contractor registry shared with `realestate_mai` maintenance contractors
- Construction contract milestones integrate with `realestate_acc` payment postings
- Management contracts link to `realestate_prm` management model configuration
- Retention balances tracked in `realestate_acc` as liability accounts
- Payment certificates integrate with `realestate_acc` AP posting workflow
- NATS events consumed by `ntfy` for SMS/email alerts to site teams
- PC certificate issuance triggers retention release workflow in `realestate_acc`

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Real-Time Defect Snagging with Photo AI Triage** [AI/Quality Assurance]
- **I2. Critical-Path Schedule Engine with Float Monitoring** [Project Management]
- **I3. Earned Value Management (EVM) Dashboard** [Cost Control / Analytics]
- **I4. Automated NEC/JBCC/FIDIC Contract Clause Compliance Checker** [Legal / Compliance]
- **I5. Subcontractor Work Package & Back-to-Back Contract Linking** [Supply Chain / Subcontracting]
- **I6. Contractor Performance Scorecard with Weighted KPIs** [Contractor Management]
- **I7. Payment Certificate Workflow with Cashflow Forecasting** [Financial Management]
- **I8. Defect Liability Period (DLP) Tracker with Automated Closure** [Post-Completion / Quality]
- **I9. NATS-Based Real-Time Event Stream for Construction Events** [Integration / Streaming]
- **I10. Risk Register with Monte Carlo Schedule/Cost Simulation** [Risk Management]
- **I11. BIM/IFC Document Integration with Drawing Register** [Document Management]
- **I12. Delay Analysis Engine (As-Planned vs. As-Built)** [Claims Management]
- **I13. Automated Quantity Surveying (QS) Cost Benchmarking** [Cost Management / AI]
- **I14. Multi-Party Notice Management with Tracked Delivery** [Legal / Notices]
- **I15. Snagging-to-Handover Digital Certificate Workflow** [Handover / Commissioning]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
