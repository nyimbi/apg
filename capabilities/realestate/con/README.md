# Property Contracts

## Overview
Full contract lifecycle management for all real estate agreements: sale/purchase, management contracts, construction contracts, service agreements, joint ventures, and development agreements. Covers party management, digital signatures, milestone tracking, variation orders (with board-approval thresholds), dispute resolution, retention management, and a searchable clause library.

## Capability ID
`realestate_con`

## Provides
- `contract_lifecycle_management`: Draft through execution, suspension, termination
- `contractor_registry_management`: Graded contractor panel (preferred to blacklisted)
- `milestone_tracking_workflow`: Typed milestones with due-date alerts and evidence capture
- `variation_order_management`: Scope/price/timeline variations with board-threshold gate
- `dispute_resolution_workflow`: Typed disputes with legal review and resolution tracking
- `contract_clause_library`: Searchable standard and custom clauses by type and tag
- `retention_management`: Percentage/fixed retention with defect-liability-clearance gate
- `contract_expiry_alerts`: Rolling expiry pipeline with configurable advance warning
- `digital_signature_workflow`: Multi-party signature tracking with method support
- `contract_performance_reporting`: Value, milestone, variation, and dispute KPIs

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | Signing authority and legal review |
| `audl` | Immutable audit for all contract events |
| `mten` | Tenant isolation |
| `conf` | Board-approval threshold configuration |
| `ntfy` | Expiry alerts, milestone overdue notifications |
| `wflo` | Board approval for large variations |
| `nlpc` | Clause library semantic search |
| `comp` | Construction law compliance |
| `mqeb` | Publish contract lifecycle events |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| `variations.board_approval_threshold` | 500,000 | KES amount requiring board approval |
| `retention.default_percentage` | 5.0 | Default retention % of contract value |
| `contractors.grading_review_months` | 12 | Months between contractor grade reviews |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/con/contracts` | GET/POST | List/create contracts | `contracts` |
| `/realestate/con/contracts/<id>/execute` | POST | Execute contract | `contracts` |
| `/realestate/con/contracts/<id>/terminate` | POST | Terminate contract | `contracts` |
| `/realestate/con/contracts/<id>/sign/<party>` | POST | Record signature | `contracts` |
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

## Data Models
- `ContractCreate/Response` — full contract with typed parties, governing law, value
- `ContractParty` — party with role, signature method, and signed-at timestamp
- `ContractorCreate/Response` — graded contractor with insurance and performance score
- `MilestoneCreate/Response` — typed milestone with due date, amount, evidence list
- `VariationOrderCreate/Response` — variation with amount/timeline change and board status
- `DisputeCreate/Response` — typed dispute with legal review flag and resolution summary
- `RetentionCreate/Response` — retention by method with defect liability end date
- `ClauseCreate/Response` — clause with type, tags, and usage count

## Streaming Events
- `contract_created`, `contract_executed`, `contract_suspended`, `contract_terminated`
- `contractor_registered`, `contractor_graded`
- `milestone_reached`, `milestone_overdue`
- `variation_raised`, `variation_approved`, `variation_rejected`
- `dispute_raised`, `dispute_resolved`, `retention_released`

## Edge Cases Handled
- Contract with only one party rejected at Pydantic model level
- Execution blocked until all party signatures are recorded AND legal review complete
- Variation against draft contract blocked (must be active)
- Board approval tracked separately from standard approval on variation
- Retention release requires both defect clearance AND approval
- Blacklisted contractor engagement denied even if contractor is in registry

## Composability Notes
- Contractor registry shared with `realestate_mai` maintenance contractors
- Construction contract milestones integrate with `realestate_acc` payment postings
- Management contracts link to `realestate_prm` management model configuration
- Retention balances tracked in `realestate_acc` as liability accounts
