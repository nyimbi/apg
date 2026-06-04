# Energy Billing & Tariffs

## Overview
Energy Billing & Tariffs manages the complete revenue cycle from tariff configuration through bill generation, payment processing, credit issuance, dispute resolution, and revenue assurance. It supports 13 tariff structures including time-of-use, demand charges, and net metering. Collection rates, write-off approvals, and revenue assurance flagging ensure financial governance across all customer classes.

## Capability ID
`energy_bil`

## Provides
| Service | Description |
|---|---|
| `tariff_management` | Create, approve and activate tariff structures per customer class |
| `consumption_billing` | Generate bills with energy and demand charges from meter readings |
| `demand_charge_calculation` | Calculate peak demand charges from interval data |
| `renewable_credits_management` | Issue, track and apply renewable energy credits |
| `revenue_assurance` | Flag and investigate unbilled energy, estimation variance and tariff errors |
| `payment_processing` | Record, reconcile and report payments across all payment methods |
| `dispute_management` | Manage billing disputes with evidence, resolution and adjusted amounts |
| `billing_analytics` | Collection rates, overdue tracking and revenue at risk reporting |

## Requires
| Capability | Reason |
|---|---|
| `auth` | User authentication and billing permissions |
| `audl` | Audit trail for tariff changes, write-offs and credits |
| `mten` | Multi-tenant billing data isolation |
| `conf` | Tariff and billing cycle configuration |
| `ntfy` | Bill issuance, overdue and dispute notifications |
| `wflo` | Tariff approval, write-off and credit issuance workflows |
| `comp` | Regulatory tariff compliance and consumer protection |
| `mqeb` | Event streaming for billing and payment lifecycle |
| `schd` | Scheduled bill generation runs |

## Configuration
| Key | Type | Default | Description |
|---|---|---|---|
| `tariffs.approval_required` | bool | true | Tariffs require approval before activation |
| `billing.auto_generate` | bool | true | Auto-generate bills at cycle end |
| `disputes.resolution_deadline_days` | int | 30 | Days to resolve a dispute |
| `payments.reconciliation_required` | bool | true | Payments must be reconciled |
| `credits.approval_required` | bool | true | Credits require approval |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| `/energy-bil/api/v1/dashboard` | GET | Dashboard with collection metrics | `energy_bil:view` |
| `/energy-bil/api/v1/tariffs` | GET | List tariffs | `energy_bil:tariffs` |
| `/energy-bil/api/v1/tariffs` | POST | Create tariff | `energy_bil:tariffs` |
| `/energy-bil/api/v1/tariffs/<id>/approve` | PUT | Approve tariff | `energy_bil:tariffs` |
| `/energy-bil/api/v1/tariffs/<id>/activate` | PUT | Activate tariff | `energy_bil:tariffs` |
| `/energy-bil/api/v1/bills` | GET | List bills | `energy_bil:billing` |
| `/energy-bil/api/v1/bills` | POST | Generate bill | `energy_bil:billing` |
| `/energy-bil/api/v1/bills/<id>/issue` | PUT | Issue bill to customer | `energy_bil:billing` |
| `/energy-bil/api/v1/bills/<id>/write-off` | PUT | Write off bill | `energy_bil:billing` |
| `/energy-bil/api/v1/payments` | POST | Record payment | `energy_bil:payments` |
| `/energy-bil/api/v1/credits` | POST | Issue credit | `energy_bil:credits` |
| `/energy-bil/api/v1/disputes` | POST | Open dispute | `energy_bil:disputes` |
| `/energy-bil/api/v1/disputes/<id>/resolve` | PUT | Resolve dispute | `energy_bil:disputes` |
| `/energy-bil/api/v1/revenue-assurance` | GET | Revenue assurance flags | `energy_bil:revenue_assurance` |
| `/energy-bil/api/v1/revenue-assurance` | POST | Flag revenue issue | `energy_bil:revenue_assurance` |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| `tenant_context_required` | tenant_context_present=False | deny |
| `tariff_type_supported` | tariff_type not in supported list | deny |
| `tariff_effective_date_required` | effective_date_present=False | deny |
| `tariff_approval_required` | activate without approval | deny |
| `bill_tariff_exists` | tariff_id not found | deny |
| `bill_meter_reading_required` | meter_reading_present=False | deny |
| `payment_amount_positive` | amount <= 0 | deny |
| `credit_approval_required` | approval_present=False | deny |
| `credit_expiry_required` | expiry_present=False | deny |
| `write_off_approval_required` | approval_present=False | deny |
| `dispute_evidence_required` | evidence_present=False | deny |
| `cross_tenant_denied` | cross_tenant_access=True | deny |
| `privileged_bil_agent_requires_human_approval` | agent write-off or credit without human approval | deny |

## Data Models
| Model | Key Fields |
|---|---|
| `Tariff` | id, name, tariff_type, customer_class, effective_date, status, rate_blocks |
| `EnergyBill` | id, customer_id, meter_id, tariff_id, billing_cycle, total_amount, status, charges |
| `Payment` | id, bill_id, payment_method, amount, reconciled, transaction_reference |
| `EnergyCredit` | id, customer_id, credit_type, amount, expires_at, approved_by, status |
| `BillingDispute` | id, bill_id, status, reason, evidence_reference, adjusted_amount |
| `RevenueAssuranceFlag` | id, flag_type, entity_id, estimated_revenue_impact, status |
| `BilAgent` | id, name, runtime, role, scope |

## Streaming Events
- `tariff_created` / `tariff_approved` / `tariff_activated`
- `bill_generated` / `bill_issued`
- `payment_received` / `payment_reconciled`
- `credit_applied`
- `dispute_opened` / `dispute_resolved`
- `revenue_assurance_flag_raised`

## Edge Cases Handled
- Bill status auto-transitions to `paid` / `partially_paid` as payments accumulate
- Tariff activation blocked if approved_by is empty
- Write-off requires separate approval from bill generation
- Credit expiry date mandatory to prevent open-ended liabilities
- Dispute evidence required at opening — not just at resolution
- Revenue assurance flags track estimated impact before investigation starts

## Composability Notes
- Receives interval readings from `energy_met` for consumption billing
- Receives REC and carbon credit data from `energy_ren` for green tariff credits
- Market settlement data from `energy_grd` feeds wholesale billing
- Dispute escalations can invoke `wflo` multi-step approval
- Revenue assurance flags feed `intel` for fraud and loss detection
