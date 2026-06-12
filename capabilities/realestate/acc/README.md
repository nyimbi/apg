# Real Estate Accounting

## Overview
Provides the full property accounting stack: chart-of-accounts management, journal entry posting with period controls, service charge raising and approval, CAM (Common Area Maintenance) reconciliation, IFRS 16 lease liability and right-of-use asset schedules, revenue recognition under multiple methods, dual-control period close, and tenant account statements.

## Capability ID
`realestate_acc`

## Provides
- `property_ledger_management`: Multi-property chart of accounts
- `service_charge_accounting`: Raise, approve, and post service charges
- `cam_reconciliation_workflow`: Estimate vs. actual CAM with lease-proportional settlement
- `cam_waterfall_allocation`: Distribute CAM variance to leases by NLA, gross area, or occupancy days
- `ifrs16_lease_accounting`: PV-based ROU asset and lease liability schedules
- `ifrs16_lease_modification`: Remeasure IFRS 16 schedule on modification with delta P&L entry
- `lease_incentive_amortisation`: Straight-line amortisation of rent-free and fit-out incentives
- `revenue_recognition_engine`: Straight-line, escalation-linked, percentage-rent (natural/artificial breakpoint), hybrid
- `journal_entry_management`: Balanced manual/automatic/reversing journals with approval
- `period_close_workflow`: Dual-control period open/close with gated checklist
- `period_close_checklist`: Pre-close validation of journals, charges, and CAM status
- `tenant_statement_generation`: Per-lease account statements
- `service_charge_dispute_workflow`: Dispute raise, review, credit-note issuance, and reversal
- `tax_calculation_engine`: VAT, withholding tax, stamp duty
- `budget_variance_reporting`: Line-level budget vs. actual comparison with tolerance flagging
- `financial_report_generation`: Trial balance, income statement, variance reports

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | User identity for approvals and audit |
| `audl` | Immutable audit trail for every posting |
| `mten` | Multi-tenant data isolation |
| `conf` | Chart-of-accounts and VAT configuration |
| `ntfy` | Alert on period close, CAM settlement |
| `wflo` | Approval workflow for journals and CAM |
| `comp` | IFRS 16 / tax compliance guardrails |
| `mqeb` | Publish accounting events to stream |
| `schd` | Auto-generate recurring journals |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| `journals.approval_required_above_amount` | 50000 | KES threshold requiring approval |
| `service_charges.cam_methods` | pro_rata | Supported CAM allocation methods |
| `ifrs16.discount_rate_required` | true | Mandate discount rate for IFRS 16 |
| `governance.dual_control_for_period_close` | true | Two approvers for period close |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/acc/dashboard` | GET | Financial summary | `view` |
| `/realestate/acc/accounts` | GET/POST | List/create accounts | `ledger` |
| `/realestate/acc/accounts/<id>` | GET/PUT | Get/update account | `ledger` |
| `/realestate/acc/journals` | GET/POST | List/create journal entries | `journals` |
| `/realestate/acc/journals/<id>/approve` | POST | Approve journal | `journals` |
| `/realestate/acc/journals/<id>/post` | POST | Post to ledger | `journals` |
| `/realestate/acc/journals/<id>/reverse` | POST | Create reversal | `journals` |
| `/realestate/acc/service-charges` | GET/POST | Service charges | `service_charges` |
| `/realestate/acc/service-charges/<id>/dispute` | POST | Raise dispute | `service_charges` |
| `/realestate/acc/disputes/<id>/credit-note` | POST | Issue credit note | `service_charges` |
| `/realestate/acc/cam` | GET/POST | CAM reconciliations | `cam` |
| `/realestate/acc/cam/<id>/settle` | POST | Settle CAM | `cam` |
| `/realestate/acc/cam/<id>/allocate` | POST | Allocate CAM variance to leases | `cam` |
| `/realestate/acc/ifrs16` | POST | Generate IFRS 16 schedule | `ifrs16` |
| `/realestate/acc/ifrs16/<id>/remeasure` | POST | Remeasure on modification | `ifrs16` |
| `/realestate/acc/lease-incentives/amortise` | POST | Amortise lease incentive | `ifrs16` |
| `/realestate/acc/revenue/percentage-rent` | POST | Recognise percentage rent | `revenue` |
| `/realestate/acc/periods` | POST | Open period | `period_close` |
| `/realestate/acc/periods/<id>/checklist` | GET | Period close checklist | `period_close` |
| `/realestate/acc/periods/<id>/close` | POST | Close period (dual control) | `period_close` |
| `/realestate/acc/statements` | POST | Generate tenant statement | `statements` |
| `/realestate/acc/reports/trial-balance` | GET | Trial balance | `reports` |
| `/realestate/acc/reports/budget-variance` | GET | Budget vs. actual variance | `reports` |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `journal_requires_balanced_entries` | debit ≠ credit | deny |
| `journal_requires_period_open` | period closed | deny |
| `journal_above_threshold_requires_approval` | amount > 50,000 & not approved | deny |
| `cam_requires_lease_links` | no leases linked | deny |
| `cam_approval_required_before_settlement` | not approved | deny |
| `period_close_requires_dual_control` | same approver | deny |
| `period_close_requires_checklist_complete` | blocking items incomplete | deny (unless force_close + third approver) |
| `delete_posted_journal_denied` | status = posted | deny |
| `ifrs16_requires_discount_rate` | no rate | deny |
| `ifrs16_remeasurement_requires_modification_date` | no modification date | deny |
| `dispute_only_on_posted_charge` | charge not posted | deny |
| `credit_note_only_on_raised_dispute` | dispute not in raised status | deny |
| `cross_tenant_posting_denied` | cross-tenant | deny |

## Data Models
- `AccountCreate/Response` — chart-of-accounts entry with type, code, ledger type
- `JournalEntryCreate/Response` — balanced debit/credit lines with period and posting status
- `ServiceChargeCreate/Response` — charge with VAT calculation and approval lifecycle
- `CamReconciliationCreate/Response` — estimated vs. actual costs with variance and settlement
- `Ifrs16ScheduleCreate/Response` — ROU asset, lease liability, amortisation schedule (with modification log)
- `RevenueScheduleCreate/Response` — revenue recognition over lease term
- `AccountingPeriodCreate/Response` — open/close with dual-control
- `TenantStatementCreate/Response` — per-lease statement with charges and payments
- `Dispute` (dict) — service charge dispute with status lifecycle
- `CreditNote` (dict) — credit note linked to dispute and original charge

## Streaming Events
- `journal_entry_created`, `journal_entry_posted`, `journal_entry_reversed`
- `service_charge_raised`, `service_charge_approved`, `service_charge_posted`
- `service_charge_disputed`, `credit_note_issued`
- `cam_reconciliation_started`, `cam_reconciliation_approved`, `cam_reconciliation_settled`
- `cam_allocated_to_leases`
- `ifrs16_schedule_generated`, `ifrs16_lease_remeasured`, `lease_incentive_amortised`
- `revenue_recognised`, `percentage_rent_recognised`
- `period_opened`, `period_closed`, `tenant_statement_generated`
- `budget_variance_report_generated`

## Edge Cases Handled
- Balanced journal enforcement at Pydantic validation time (not just service layer)
- Reversal automatically flips debit/credit lines from original
- IFRS 16 monthly-rate conversion uses exact payment-per-month division
- IFRS 16 remeasurement uses modification-date PV; stores delta for P&L entry
- Lease incentive amortisation clamps accumulated amount to total incentive (no over-amortisation)
- Percentage-rent natural breakpoint computed as `base_rent / rate`; artificial breakpoint supplied explicitly
- CAM waterfall allocation asserts total basis > 0 to prevent division by zero
- CAM settlement blocked until reconciliation is in `approved` status
- Period close blocked if any journal/charge/CAM item is outstanding (checklist gating)
- Dispute creation only permitted on `posted` charges; credit note only on `raised` disputes
- Budget variance report handles missing budget (returns zero budget totals gracefully)
- Duplicate period open prevented via store scan
- Tax calculation returns both tax amount and gross amount

## Composability Notes
- Feeds into `realestate_ren` (rent collection postings) and `realestate_lea` (IFRS 16 schedules)
- Consumes lease data from `realestate_lea` for revenue schedules
- CAM reconciliation integrates with `realestate_prm` for property-level costs
- IFRS 16 schedules are referenced by `realestate_lea` for balance sheet reporting

---

## World-Class Enhancements (v2.0)

1. **Waterfall CAM Allocation Engine** — Distribute CAM variance to leases by NLA, gross area, fixed %, or weighted occupancy days via `allocate_cam_to_leases()`.
2. **IFRS 16 Lease Modification Handling** — Atomic remeasurement on rent review/extension/surrender with delta P&L journal via `remeasure_ifrs16_lease()`.
3. **Rent-Free Period Amortisation** — Monthly straight-line amortisation of lease incentives with deferred incentive journal auto-generation via `amortise_lease_incentive()`.
4. **Percentage-Rent (Turnover Rent) Recognition** — Natural and artificial breakpoint calculation posting variable component as a separate journal line via `recognise_percentage_rent()`.
5. **Multi-Currency Revaluation and FX Gains/Losses** — IAS 21-compliant spot-rate revaluation of foreign-currency balances with unrealised FX P&L journals via `record_fx_revaluation()`.
6. **Operating Expense Accruals with Auto-Reversal** — Accrual journals with `reversal_date` field; `schd`-integrated auto-reversal on period-start via `accrue_operating_expense()`.
7. **Sinking Fund (Reserve Fund) Management** — Per-property reserve ledger with NLA-proportional tenant contributions and fund adequacy reporting via `create_sinking_fund()` / `record_sinking_fund_contribution()`.
8. **Audit-Trail Immutable Ledger Integration** — Every write publishes a structured `AuditEvent` to `audl` with before/after JSON Patch diff, actor, and session context.
9. **Budget vs. Actual Variance Reporting** — Line-level and total variance with configurable tolerance flagging consumable by `realestate_rep` via `budget_variance_report()`.
10. **Withholding Tax (WHT) Compliance Workflow** — Full KRA WHT lifecycle: certificate generation, M-payment remittance scheduling, and iTax reconciliation via `generate_wht_certificate()` / `schedule_wht_remittance()` / `reconcile_wht_with_revenue_authority()`.
11. **Property Disposal and Derecognition** — IAS 40/IFRS 5 disposal: gain/loss computation, asset derecognition, IFRS 16 schedule closure, composite disposal journal in one transaction via `record_property_disposal()`.
12. **Service Charge Dispute and Credit Note Workflow** — Three-state dispute workflow (raised → under_review → resolved/rejected) with auto-reversal journals and `ntfy` notification via `raise_service_charge_dispute()` / `issue_credit_note()`.
13. **Lease Incentive Liability (Lessor Perspective)** — Deferred income liability for landlord rent-free grants amortised to P&L over lease term via `schd`-scheduled journals via `record_lessor_lease_incentive()`.
14. **Cash Flow Statement Preparation** — IAS 7-compliant operating/investing/financing classification using `ledger_type` with bank reconciliation and unclassified-account flagging via `prepare_cash_flow_statement()`.
15. **Automated Period-End Checklist and Close Gating** — Structured pre-close checklist (journals, charges, CAM, WHT) gates `close_period()`; unblocked only by `force_close` + third approver via `get_period_close_checklist()`.

---

## New Methods

### `allocate_cam_to_leases` — CAM variance distribution

Most impactful for multi-tenant properties where CAM reconciliation variance must be pushed back to individual lease statements.

```python
result = await svc.allocate_cam_to_leases(
    cam_id="cam_01j...",
    tenant_id="t_01j...",
    allocation_basis="nla",          # "nla" | "gross_area" | "fixed_pct" | "occupancy_days"
    lease_nla_map={"lea_01": 450.0, "lea_02": 230.0, "lea_03": 320.0},
)
# result["allocations"] — per-lease credit/debit amounts
# result["total_variance_allocated"] — must equal CAM variance
# emits: cam_allocated_to_leases
```

### `remeasure_ifrs16_lease` — Lease modification remeasurement

Handles mid-term rent reviews and extensions without reconstructing the entire schedule from scratch.

```python
result = await svc.remeasure_ifrs16_lease(
    schedule_id="sch_01j...",
    tenant_id="t_01j...",
    modification_date="2026-07-01",
    revised_monthly_payment=Decimal("185000"),
    revised_discount_rate=Decimal("0.095"),
    remaining_term_months=36,
)
# result["delta_liability"]   — new liability minus old (posted to P&L)
# result["new_schedule"]      — appended amortisation segment
# result["journal_id"]        — remeasurement gain/loss journal reference
# emits: ifrs16_lease_remeasured
```

### `get_period_close_checklist` + `close_period` — Gated period close

Inspect checklist before attempting close; `close_period` raises if any blocking item is incomplete.

```python
checklist = await svc.get_period_close_checklist(period_id="per_01j...", tenant_id="t_01j...")
# checklist["items"] — list of {name, status, blocking}
# checklist["checklist_complete"] — bool

if checklist["checklist_complete"]:
    closed = await svc.close_period(
        period_id="per_01j...",
        tenant_id="t_01j...",
        closed_by="user_a",
        second_approver="user_b",
    )
# Pass force_close=True + third_approver to override blocking items (escalated close)
```
