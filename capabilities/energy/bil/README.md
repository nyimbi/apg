# Energy Billing & Tariffs

## Overview
Energy Billing & Tariffs manages the complete revenue cycle from tariff configuration through bill generation, payment processing, credit issuance, dispute resolution, and revenue assurance. It supports 13+ tariff structures including time-of-use, demand charges, net metering, prepaid/token vending, and multi-currency billing. Collection rates, write-off approvals, and revenue assurance flagging ensure financial governance across all customer classes.

## Capability ID
`energy_bil`

## Features
- Tariff lifecycle: create / approve / activate / supersede with rate block support
- Tiered and ToU energy charge calculation with fuel adjustment pass-through
- Demand charge calculation with ratchet clause enforcement
- Regulatory levy application (REP, REREC, ERC, VAT, fuel levy)
- Net Energy Metering (NEM) credit calculation for solar prosumers
- Prepaid STS token vending with forced debt recovery
- Granular 15/30-min interval billing (TimescaleDB-backed)
- Automated bulk bill generation with cycle scheduling
- 6-tier dunning workflow with EPRA hold period enforcement
- Revenue leakage rule engine (6 built-in rules, ML-scored)
- Carbon emission billing with Scope 2 market-based reporting
- Automated meter data validation (rollover, outlier, tamper detection)
- Cross-subsidiary consolidated billing with volume discount tiers
- Multi-currency billing with FX rate snapshots at generation time
- Smart invoice rendering: PDF / HTML / JSON / UBL-XML with M-Pesa QR code
- Payment plan covenant monitoring with auto-default escalation
- EPRA tariff compliance checking at bill generation
- Predictive collection KPIs (30/60/90-day forecast, AI narrative via Ollama)
- ML-powered anomaly detection (Ollama-backed, graceful fallback)
- Payment analytics, arrears analytics, billing analytics by period

## Provides
| Service | Description |
|---|---|
| `tariff_management` | Create, approve and activate tariff structures per customer class |
| `consumption_billing` | Generate bills with energy and demand charges from meter readings |
| `demand_charge_calculation` | Calculate peak demand charges including ratchet clause |
| `renewable_credits_management` | Issue, track and apply renewable energy credits |
| `revenue_assurance` | Flag and investigate unbilled energy, estimation variance and tariff errors |
| `payment_processing` | Record, reconcile and report payments across all payment methods |
| `dispute_management` | Manage billing disputes with evidence, resolution and adjusted amounts |
| `billing_analytics` | Collection rates, overdue tracking and revenue at risk reporting |
| `nem_billing` | Net Energy Metering credits for solar prosumers |
| `prepaid_vending` | STS token generation and prepaid reconciliation |
| `dunning_workflow` | 6-tier escalation engine with EPRA-compliant hold periods |
| `carbon_billing` | Scope 2 emission calculation and reporting per bill |
| `collection_forecast` | AI-assisted 30/60/90-day collection probability by segment |

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

## Quick Start

```python
from capabilities.energy.bil.service import EnergyBillingService

svc = EnergyBillingService(tenant_id="kplc", actor_id="operator-1")

# Create and activate a residential tariff
tariff = svc.create_tariff(
    tariff_id="T-RES-01", tenant_id="kplc", name="DC1 Residential",
    tariff_type="tiered", customer_class="residential",
    effective_date="2026-01-01", created_by="tariff-admin",
    rate_blocks=[{"limit_kwh": 100, "rate": 12.0}, {"limit_kwh": None, "rate": 18.5}],
)
svc.approve_tariff("T-RES-01", "kplc", approved_by="ceo")
svc.activate_tariff("T-RES-01", "kplc")

# Generate and issue a bill
bill = svc.generate_bill(
    bill_id="B-001", tenant_id="kplc", customer_id="C-100",
    meter_id="M-555", tariff_id="T-RES-01", billing_cycle="monthly",
    period_start="2026-05-01", period_end="2026-05-31",
    consumption_kwh=320.5, peak_demand_kw=3.2,
    charges=[{"type": "energy", "amount": 4820.0}], total_amount=5603.2,
)
svc.issue_bill("B-001", "kplc", due_date="2026-06-20")

# Record an M-Pesa payment
svc.record_payment(
    payment_id="P-001", tenant_id="kplc", bill_id="B-001",
    customer_id="C-100", payment_method="mpesa",
    amount=5603.2, currency="KES", transaction_reference="QJZ8F7D1",
)

print(svc.dashboard_summary("kplc"))
```

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
| `/energy-bil/api/v1/billing-runs` | POST | Trigger bulk bill run | `energy_bil:billing` |

## New Methods

### `calculate_energy_charges` — Tiered/ToU energy pricing

```python
result = await svc.calculate_energy_charges(
    account_id="C-100", period="2026-05",
    consumption_kwh=320.5,
    rate_blocks=[
        {"limit_kwh": 100, "rate": 12.0},
        {"limit_kwh": 300, "rate": 16.5},
        {"limit_kwh": None, "rate": 21.0},
    ],
    fuel_adjustment_rate=0.85,  # KES/kWh pass-through
    tou_multiplier=1.4,         # peak-hour multiplier
)
# {"tier_breakdown": [...], "total_energy_charge": 5821.35, ...}
```

### `apply_levies` — EPRA regulatory levies

```python
levies = await svc.apply_levies(
    account_id="C-100", period="2026-05",
    levy_types=["REP", "REREC", "ERC", "VAT"],
    consumption_kwh=320.5,
    energy_charge=5821.35,
)
# {"levy_items": [{"levy_type": "VAT", "rate": 0.16, "amount": 931.42}, ...], "total_levies": 1062.55}
```

### `calculate_demand_charges` — Peak demand with ratchet

```python
demand = await svc.calculate_demand_charges(
    account_id="C-200", period="2026-05",
    peak_demand_kw=85.0,
    demand_rate_per_kw=450.0,
    ratchet_pct=75.0,  # bill at min 75% of 12-month peak
)
# {"billing_demand_kw": 85.0, "ratchet_demand_kw": 63.75, "total_demand_charge": 38250.0, ...}
```

### `billing_analytics` — Period roll-up KPIs

```python
analytics = await svc.billing_analytics("2026-05")
# {"bills_generated": 1420, "collection_rate_pct": 91.3, "total_arrears_amount": 182500.0, ...}
```

### `arrears_management` — Structured collections workflow

```python
plan = await svc.arrears_management(
    account_id="C-300",
    arrears_amount=45000.0,
    action="payment_plan",
    payment_plan_months=6,
)
# {"monthly_instalment": 7500.0, "status": "active", ...}
# Auto-raises RevenueAssuranceFlag for amounts > KES 10,000
```

## World-Class Enhancements (v2.0)

1. **Granular ToU Interval Billing** — 15/30-min DLMS/COSEM interval reads priced per on/mid/off-peak window; eliminates NEM gaming and unlocks EPRA MELT compliance.
2. **Automated Bill Run Scheduler** — `BillingCycleSchedule` + APScheduler/Celery Beat triggers `bulk_generate_bills` at cycle cut-off; failed runs journaled with retry state.
3. **Predictive Consumption Estimation (ML)** — `estimate_consumption()` uses LSTM/GBT or Ollama `llama3`/`phi3` for <2% estimation error on ~8% monthly missed reads.
4. **Multi-Currency + FX Rate Management** — `FXRateSnapshot` locks exchange rate at bill generation; blocks billing if no valid FX pair; supports KES/USD/EUR industrial contracts.
5. **Net Energy Metering (NEM) for Solar Prosumers** — `calculate_nem_credit()` nets export vs. import per interval; rollover surplus to next month per Kenya Energy Act 2019.
6. **Prepaid STS Token Vending** — `vend_token()` generates IEC 62055-41 20-digit tokens; forced debt recovery before unit allocation; `reconcile_prepaid_vending()` closes the loop.
7. **6-Tier Dunning Workflow Engine** — `advance_dunning()` enforces EPRA hold periods per tier (reminder → legal referral); auto-escalation flag per tier; reduces write-offs 15–30%.
8. **Revenue Leakage Rule Engine** — `RevenueLeakageScanner` runs 6 rules (zero-consumption, unbilled meters, orphan payments, tariff mismatch, demand shortfall, excess credits) with ML confidence.
9. **EPRA Tariff Compliance Checking** — `validate_tariff_compliance()` called as `generate_bill()` pre-condition; validates rate caps, levy presence, and customer class segmentation.
10. **Smart Invoice with QR Payment** — `render_invoice()` outputs PDF/HTML/JSON/UBL-XML with embedded M-Pesa QR code; digital signature; WhatsApp/email dispatch via `ntfy`.
11. **Payment Plan + Covenant Monitoring** — `PaymentPlan` model tracks instalments; `check_payment_plan_covenants()` auto-defaults after `max_misses` and re-escalates dunning.
12. **Carbon Emission Billing (Scope 2)** — `calculate_carbon_charges()` embeds grid emission factor per kWh; REC offset for green tariffs; carbon summary block in rendered invoice.
13. **Automated Meter Data Validation (AMDV)** — `validate_meter_reads()` detects null reads, register rollover, reverse tamper, and >3σ outliers; returns substitution estimates.
14. **Cross-Subsidiary Consolidated Billing** — `generate_consolidated_bill()` aggregates multi-site charges; applies group volume discounts; routes settlement to primary account.
15. **Real-Time Collection Dashboard + Predictive KPIs** — `collection_forecast()` computes 30/60/90-day expected collection by segment; Ollama narrative insight if available.

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
- `tariff_created` / `tariff_approved` / `tariff_activated` / `tariff_superseded`
- `bill_generated` / `bill_issued` / `bill_written_off`
- `payment_received` / `payment_reconciled`
- `credit_applied`
- `dispute_opened` / `dispute_resolved`
- `revenue_assurance_flag_raised`
- `arrears_managed` / `bills_exported` / `tariff_changed`
- `billing_compliance_report_generated`

## Edge Cases Handled
- Bill status auto-transitions to `paid` / `partially_paid` as payments accumulate
- Tariff activation blocked if `approved_by` is empty
- Write-off requires separate approval from bill generation
- Credit expiry date mandatory to prevent open-ended liabilities
- Dispute evidence required at opening — not just at resolution
- Revenue assurance flags raised automatically for arrears > KES 10,000
- `bulk_generate_bills` isolates per-account errors; partial failures reported without aborting the run
- `tariff_change` supersedes the old active tariff atomically before creating the new version
- ML methods (`ml_billing_anomaly`, `collection_forecast`) degrade gracefully when `OLLAMA_BASE_URL` is unset

## Composability Notes
- Receives interval readings from `energy_met` for consumption billing
- Receives REC and carbon credit data from `energy_ren` for green tariff credits
- Market settlement data from `energy_grd` feeds wholesale billing and carbon emission factors
- Dispute escalations can invoke `wflo` multi-step approval
- Revenue assurance flags feed `intel` for fraud and loss detection
- NEM credit data shared with `energy_ren` for prosumer portfolio reporting
