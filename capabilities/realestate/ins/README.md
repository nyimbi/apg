# Property Insurance

## Overview
End-to-end property insurance portfolio management: policy creation and binding with asset schedules, claims lodgement through settlement with large-claim senior-approval gates, endorsement issuance, premium allocation across properties, automated coverage gap detection, insurer/broker registry, and renewal pipeline tracking.

## Capability ID
`realestate_ins`

## Provides
- `policy_lifecycle_management`: Property all-risk, fire, liability, and 8 other policy types
- `asset_schedule_management`: Insured asset register linked to policies with valuation basis
- `claims_processing_workflow`: Lodge, investigate, approve (senior for large), settle
- `premium_allocation_engine`: 5 allocation methods including GLA and risk-weighted
- `coverage_gap_analysis`: Automated gap detection with critical-gap mandatory alerting
- `endorsement_management`: 7 endorsement types with sum-insured adjustment
- `insurer_broker_registry`: Graded insurer panel with suspension enforcement
- `renewal_pipeline_tracking`: 90-day advance renewal pipeline
- `insurance_reporting`: Claims frequency, premium adequacy, gap summary
- `compliance_certificate_management`: Certificate issue against active policies only

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | Senior approval for large claims |
| `audl` | Immutable claims and endorsement audit |
| `mten` | Multi-tenant policy isolation |
| `conf` | Policy type and threshold configuration |
| `ntfy` | Renewal due, critical gap, large claim alerts |
| `wflo` | Claims approval workflow |
| `comp` | Insurance regulatory compliance |
| `mqeb` | Publish insurance lifecycle events |
| `schd` | Renewal reminder scheduling |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| `claims.large_claim_threshold` | 1,000,000 | KES requiring senior approval |
| `renewals.early_warning_days` | 90 | Days before expiry for renewal alert |
| `gaps.auto_alert_on_critical` | true | Mandatory alert on critical gap |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/ins/policies` | GET/POST | List/create policies | `policies` |
| `/realestate/ins/policies/<id>/bind` | POST | Bind policy | `policies` |
| `/realestate/ins/renewals` | GET | Renewal pipeline | `renewals` |
| `/realestate/ins/assets` | GET/POST | Asset schedule | `assets` |
| `/realestate/ins/claims` | GET/POST | Claims | `claims` |
| `/realestate/ins/claims/<id>/approve` | POST | Approve claim | `claims` |
| `/realestate/ins/claims/<id>/settle` | POST | Settle claim | `claims` |
| `/realestate/ins/endorsements` | GET/POST | Endorsements | `endorsements` |
| `/realestate/ins/premiums` | POST | Allocate premium | `premiums` |
| `/realestate/ins/gaps` | GET | Coverage gaps | `gaps` |
| `/realestate/ins/gaps/detect/<property_id>` | POST | Detect gaps | `gaps` |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `suspended_insurer_cannot_bind` | insurer suspended | deny |
| `policy_requires_asset_schedule` | no assets | deny |
| `claim_requires_active_policy` | policy inactive | deny |
| `claim_peril_must_be_covered` | peril excluded | deny |
| `large_claim_requires_approval` | > 1M, no senior | deny |
| `settlement_cannot_exceed_sum_insured` | exceeds sum | deny |
| `critical_gap_triggers_mandatory_alert` | critical, no alert | deny |
| `certificate_requires_active_policy` | policy inactive | deny |

## Data Models
- `PolicyCreate/Response` — full policy with perils, deductibles, insurer, and status
- `PolicyDeductible` — typed deductible (fixed/percentage/franchise/excess)
- `InsuredAssetCreate/Response` — asset on schedule with insured value and valuation basis
- `ClaimCreate/Response` — claim with peril, incident date, estimated/approved/settled amounts
- `EndorsementCreate/Response` — typed endorsement with sum-insured and premium adjustments
- `PremiumAllocationCreate/Response` — allocation run with per-property breakdown
- `CoverageGapCreate/Response` — detected gap with severity and remediation tracking
- `InsurerCreate/Response` — graded insurer with active policy count

## Streaming Events
- `policy_created`, `policy_bound`, `policy_lapsed`, `policy_expired`
- `asset_added_to_schedule`, `asset_removed_from_schedule`
- `claim_lodged`, `claim_assessed`, `claim_approved`, `claim_settled`
- `premium_allocated`, `endorsement_issued`
- `renewal_due`, `coverage_gap_detected`, `certificate_issued`

## Edge Cases Handled
- Binding blocked if insurer is suspended, regardless of other fields
- Binding requires at least one asset on the schedule
- Claim against inactive policy rejected before peril check
- Settlement amount validated against sum insured at service layer
- Critical coverage gap generates mandatory alert event before recording
- Policy sum insured updated automatically when endorsement changes it
- Insurer active policy count incremented on policy creation

## Composability Notes
- Insurance premiums posted to `realestate_acc` as opex charges
- Asset schedule references `realestate_prm` property and asset register
- Coverage gap detection triggers `ntfy` alerts to property managers
- Reinstatement cost valuations link to `realestate_val` assessment
