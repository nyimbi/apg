# Real Estate Accounting

## Overview
Provides the full property accounting stack: chart-of-accounts management, journal entry posting with period controls, service charge raising and approval, CAM (Common Area Maintenance) reconciliation, IFRS 16 lease liability and right-of-use asset schedules, revenue recognition under multiple methods, dual-control period close, and tenant account statements.

## Capability ID
`realestate_acc`

## Provides
- `property_ledger_management`: Multi-property chart of accounts
- `service_charge_accounting`: Raise, approve, and post service charges
- `cam_reconciliation_workflow`: Estimate vs. actual CAM with lease-proportional settlement
- `ifrs16_lease_accounting`: PV-based ROU asset and lease liability schedules
- `revenue_recognition_engine`: Straight-line, escalation-linked, percentage-rent, hybrid
- `journal_entry_management`: Balanced manual/automatic/reversing journals with approval
- `period_close_workflow`: Dual-control period open/close with reconciliation gating
- `tenant_statement_generation`: Per-lease account statements
- `tax_calculation_engine`: VAT, withholding tax, stamp duty
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
| `/realestate/acc/cam` | GET/POST | CAM reconciliations | `cam` |
| `/realestate/acc/cam/<id>/settle` | POST | Settle CAM | `cam` |
| `/realestate/acc/ifrs16` | POST | Generate IFRS 16 schedule | `ifrs16` |
| `/realestate/acc/periods` | POST | Open period | `period_close` |
| `/realestate/acc/periods/<id>/close` | POST | Close period (dual control) | `period_close` |
| `/realestate/acc/statements` | POST | Generate tenant statement | `statements` |
| `/realestate/acc/reports/trial-balance` | GET | Trial balance | `reports` |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `journal_requires_balanced_entries` | debit ≠ credit | deny |
| `journal_requires_period_open` | period closed | deny |
| `journal_above_threshold_requires_approval` | amount > 50,000 & not approved | deny |
| `cam_requires_lease_links` | no leases linked | deny |
| `cam_approval_required_before_settlement` | not approved | deny |
| `period_close_requires_dual_control` | same approver | deny |
| `delete_posted_journal_denied` | status = posted | deny |
| `ifrs16_requires_discount_rate` | no rate | deny |
| `cross_tenant_posting_denied` | cross-tenant | deny |

## Data Models
- `AccountCreate/Response` — chart-of-accounts entry with type, code, ledger type
- `JournalEntryCreate/Response` — balanced debit/credit lines with period and posting status
- `ServiceChargeCreate/Response` — charge with VAT calculation and approval lifecycle
- `CamReconciliationCreate/Response` — estimated vs. actual costs with variance and settlement
- `Ifrs16ScheduleCreate/Response` — ROU asset, lease liability, amortisation schedule
- `RevenueScheduleCreate/Response` — revenue recognition over lease term
- `AccountingPeriodCreate/Response` — open/close with dual-control
- `TenantStatementCreate/Response` — per-lease statement with charges and payments

## Streaming Events
- `journal_entry_created`, `journal_entry_posted`, `journal_entry_reversed`
- `service_charge_raised`, `service_charge_approved`, `service_charge_posted`
- `cam_reconciliation_started`, `cam_reconciliation_approved`, `cam_reconciliation_settled`
- `ifrs16_schedule_generated`, `revenue_recognised`
- `period_opened`, `period_closed`, `tenant_statement_generated`

## Edge Cases Handled
- Balanced journal enforcement at Pydantic validation time (not just service layer)
- Reversal automatically flips debit/credit lines from original
- IFRS 16 monthly-rate conversion uses exact payment-per-month division
- CAM settlement blocked until reconciliation is in `approved` status
- Duplicate period open prevented via store scan
- Tax calculation returns both tax amount and gross amount

## Composability Notes
- Feeds into `realestate_ren` (rent collection postings) and `realestate_lea` (IFRS 16 schedules)
- Consumes lease data from `realestate_lea` for revenue schedules
- CAM reconciliation integrates with `realestate_prm` for property-level costs
- IFRS 16 schedules are referenced by `realestate_lea` for balance sheet reporting
