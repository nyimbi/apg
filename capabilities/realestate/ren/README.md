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

## New Capabilities (v1.1)

### Inspection Workflow
| Method | Description |
|--------|-------------|
| `record_inspection()` | Move-in / mid-term / move-out condition grading with photo IDs |

### Rent Review
| Method | Description |
|--------|-------------|
| `propose_rent_increase()` | Statutory-notice-aware rent increase proposal |
| `apply_rent_increase()` | Commit increase after effective_date passes |

### Void Tracking
| Method | Description |
|--------|-------------|
| `record_void_period()` | Log gap between tenancies with reason code |
| `get_void_report()` | Void rate % and breakdown by reason |

### Rent Roll Snapshots
| Method | Description |
|--------|-------------|
| `snapshot_rent_roll()` | Named point-in-time rent roll snapshot |
| `compare_rent_rolls()` | Diff two snapshots: added / removed / changed tenancies |

### Statements & Receipts
| Method | Description |
|--------|-------------|
| `get_tenancy_statement()` | Chronological charge/payment ledger with running balance |
| `generate_rent_receipt()` | Formal `REC-YYYY-NNNN` receipt per payment |

### Arrears Automation
| Method | Description |
|--------|-------------|
| `schedule_arrears_chase()` | Multi-step chase schedule (email / SMS / letter / phone) |

### Compliance
| Method | Description |
|--------|-------------|
| `run_compliance_check()` | Checklist engine: deposit, referencing, EPC, gas cert, EICR |

## New API Routes (v1.1)
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/ren/tenancies/<id>/statement` | GET | Statement of account | `tenancies` |
| `/realestate/ren/tenancies/<id>/rent-increase` | POST | Propose rent increase | `tenancies` |
| `/realestate/ren/inspections` | GET/POST | Inspections | `inspections` |
| `/realestate/ren/voids` | GET/POST | Void periods | `voids` |
| `/realestate/ren/voids/report` | GET | Void rate report | `voids` |
| `/realestate/ren/rent-roll/snapshot` | POST | Snapshot rent roll | `rent_roll` |
| `/realestate/ren/rent-roll/compare` | GET | Compare snapshots | `rent_roll` |
| `/realestate/ren/payments/<id>/receipt` | POST | Generate receipt | `rent_collection` |
| `/realestate/ren/arrears/<id>/chase` | POST | Schedule arrears chase | `arrears` |
| `/realestate/ren/compliance/<tenancy_id>` | GET | Run compliance check | `compliance` |

## New Streaming Events (v1.1)
- `inspection_recorded`, `inspection_move_out`
- `rent_increase_proposed`, `rent_increase_applied`
- `void_opened`, `void_closed`
- `rent_roll_snapshot_created`
- `receipt_issued`
- `chase_scheduled`, `chase_step_sent`
- `compliance_check_completed`, `compliance_failed`

## World-Class Enhancements (v2.0)

1. **Persistent SQL Store via AsyncPG** — swap in-memory dict for AsyncPG connection pool; zero service-layer refactor required.
2. **Rent-Increase Workflow** — `propose_rent_increase()` enforces statutory notice period and blocks application until `effective_date`.
3. **Move-In / Move-Out Inspection Workflow** — structured condition grading with photo evidence IDs; dispute-proof deposit deductions.
4. **Automated Arrears Chasing Schedule** — `schedule_arrears_chase()` fires multi-step sequences via `schd` capability; no manual follow-up.
5. **Multi-Currency Rent Collection with FX Conversion** — pluggable `FXProvider` converts to base currency; records `currency_gain_loss` per payment.
6. **Rent Receipt Generation (PDF)** — `generate_rent_receipt()` produces `REC-YYYY-NNNN` sequential receipts; meets legal requirements.
7. **Vacancy Tracking and Void Analysis** — `record_void_period()` and `get_void_report()` surface `void_rate_pct` in analytics.
8. **Statement of Account per Tenancy** — `get_tenancy_statement()` returns chronological ledger with opening/closing balance and days-in-arrears.
9. **Guarantor Management** — structured `GuarantorCreate/Response` models with `guarantee_limit`, expiry, and `call_on_guarantor()` linked to arrears escalation.
10. **Lease Break Clause Tracking** — `BreakClause` sub-model with `exercise_break_clause()` validating conditions and transitioning tenancy to `vacating`.
11. **Concurrent Modification Guard (Optimistic Locking)** — `updated_at` version check on all mutations; raises `ConflictError` on stale-write.
12. **Partial Payment Allocation (FIFO)** — `allocate_payment()` clears oldest arrears first; produces per-period `AllocationResult`.
13. **Regulatory Compliance Checklist Engine** — `run_compliance_check()` returns per-item pass/fail for gas cert, EICR, EPC, fire safety, deposit protection.
14. **Webhook / Event Bus Emission** — injected `EventEmitter` adapter calls `await self._emit(event_type, payload)` post-mutation; wires to `mqeb` in production.
15. **Rent Roll Versioned Snapshots** — `snapshot_rent_roll()` and `compare_rent_rolls()` provide month-end reconciliation and auditor evidence packs.

## New Methods

### `get_tenancy_statement()` — unified ledger for self-service portal

```python
statement = await svc.get_tenancy_statement(
    tenancy_id="ten-001",
    tenant_id="t-acme",
    from_date=date(2025, 1, 1),
    to_date=date(2025, 6, 30),
)
# Returns: opening_balance, list of {date, type, amount, running_balance},
#          closing_balance, days_in_arrears
print(statement["closing_balance"], statement["days_in_arrears"])
```

### `schedule_arrears_chase()` — automated multi-step collection

```python
chase = await svc.schedule_arrears_chase(
    arrears_id="arr-042",
    chase_sequence=[
        {"days_after": 3,  "method": "sms"},
        {"days_after": 7,  "method": "email"},
        {"days_after": 14, "method": "letter"},
        {"days_after": 30, "method": "phone"},
    ],
    tenant_id="t-acme",
)
# Enqueues steps via `schd`; fires `chase_scheduled` event.
# chase["steps_scheduled"] == 4
```

### `compare_rent_rolls()` — month-end reconciliation diff

```python
snap_march = await svc.snapshot_rent_roll(tenant_id="t-acme", snapshot_date=date(2025, 3, 31))
snap_april = await svc.snapshot_rent_roll(tenant_id="t-acme", snapshot_date=date(2025, 4, 30))

diff = await svc.compare_rent_rolls(
    id_a=snap_march["snapshot_id"],
    id_b=snap_april["snapshot_id"],
    tenant_id="t-acme",
)
# diff keys: "added", "removed", "changed" — each a list of tenancy diffs
print(f"+{len(diff['added'])} tenancies, -{len(diff['removed'])}, ~{len(diff['changed'])} changed")
```
