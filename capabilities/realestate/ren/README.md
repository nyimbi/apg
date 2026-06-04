# Rental Operations

## Overview
End-to-end tenancy lifecycle: application, referencing, right-to-rent checks, deposit registration and accounting, rent collection with shortfall detection, arrears management and legal escalation, notice serving, and renewal pipeline management. Produces a live rent roll for any property.

## Capability ID
`realestate_ren`

## Provides
- `tenancy_lifecycle_management`: Application through vacating with status-driven workflow
- `rent_collection_engine`: Payment recording with shortfall detection and arrears flagging
- `arrears_management_workflow`: Automated ageing buckets and legal escalation gating
- `deposit_accounting`: Cash and insurance deposits with evidence-gated deductions
- `tenancy_renewal_pipeline`: Renewal offer, acceptance, and new-term generation
- `referencing_workflow`: Credit, employment, landlord, and right-to-rent checks
- `notice_management`: Section 21, Section 8, quit notice, break notice
- `legal_action_tracking`: LBA through charging order with arrears threshold gate
- `rent_roll_management`: Live rent roll with arrears status per tenancy
- `tenancy_performance_reporting`: Void rate, collection rate, arrears aging

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | Approval for legal actions |
| `audl` | Audit trail for payments and notices |
| `mten` | Tenant isolation |
| `conf` | Tenancy type and rent frequency configuration |
| `ntfy` | Rent overdue, renewal due, notice served alerts |
| `wflo` | Legal escalation approval |
| `comp` | Right-to-rent compliance |
| `mqeb` | Publish rental events |
| `schd` | Renewal reminder scheduling |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| `arrears.legal_threshold_days` | 90 | Days overdue before legal action allowed |
| `deposits.registration_required` | true | Mandate deposit scheme registration |
| `renewals.early_warning_days` | 90 | Days before expiry to trigger renewal |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/ren/tenancies` | GET/POST | List/create tenancies | `tenancies` |
| `/realestate/ren/tenancies/<id>/activate` | POST | Activate tenancy | `tenancies` |
| `/realestate/ren/rent-collection` | GET/POST | Payments | `rent_collection` |
| `/realestate/ren/arrears` | GET | Active arrears | `arrears` |
| `/realestate/ren/arrears/<id>/legal` | POST | Escalate to legal | `legal` |
| `/realestate/ren/deposits` | POST | Register deposit | `deposits` |
| `/realestate/ren/deposits/<id>/deduct` | POST | Deduct (evidence req.) | `deposits` |
| `/realestate/ren/deposits/<id>/release` | POST | Release deposit | `deposits` |
| `/realestate/ren/notices` | GET/POST | Notices | `notices` |
| `/realestate/ren/renewals` | POST | Initiate renewal | `renewals` |
| `/realestate/ren/renewals/pipeline` | GET | Renewal pipeline | `renewals` |
| `/realestate/ren/referencing` | POST | Run referencing | `referencing` |
| `/realestate/ren/rent-roll` | GET | Rent roll | `rent_roll` |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `activation_requires_deposit_registered` | not registered | deny |
| `activation_requires_referencing_complete` | not complete | deny |
| `right_to_rent_required_for_residential` | AST, not checked | deny |
| `deposit_deduction_requires_evidence` | no evidence docs | deny |
| `deposit_deduction_cannot_exceed_held` | exceeds balance | deny |
| `legal_action_requires_arrears_threshold` | < 90 days | deny |
| `vacated_tenancy_modification_restricted` | status = vacated | deny |

## Data Models
- `TenancyCreate/Response` — full tenancy header with rent, frequency, and pre-condition flags
- `RentPaymentCreate/Response` — payment with shortfall and receipt number
- `ArrearsRecordCreate/Response` — overdue amount with aging bucket status
- `DepositCreate/Response` — deposit type, scheme reference, deduction history
- `DepositDeductionCreate/Response` — itemised deduction with evidence document links
- `NoticeCreate/Response` — notice type, served date, effective date
- `TenancyRenewalCreate/Response` — new terms offer with acceptance tracking
- `ReferencingCreate/Response` — check types, results, pass/fail status

## Streaming Events
- `tenancy_created`, `tenancy_activated`, `tenancy_vacated`
- `rent_received`, `rent_overdue`, `arrears_escalated`
- `deposit_registered`, `deposit_released`, `deposit_disputed`
- `renewal_initiated`, `renewal_completed`
- `notice_served`, `legal_action_commenced`

## Edge Cases Handled
- Short payment auto-creates arrears record without explicit call
- Full payment clears existing arrears status
- AST right-to-rent check enforced by tenancy_type at activation
- Deposit deduction exceeding held balance rejected at service layer
- Legal escalation gated by days_overdue >= 90 AND amount > 0
- Vacated tenancies render immutable to prevent stale state modification

## Composability Notes
- Consumes lease terms from `realestate_lea` for expected rent amounts
- Posts rent receipts to `realestate_acc` journal entries
- Unit availability updates feed back to `realestate_prm`
- Tenant communication integrates with `realestate_ten` portal
