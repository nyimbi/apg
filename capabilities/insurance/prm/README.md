# Premium & Billing (ins_prm)

Premium calculation, instalment management, collections, reconciliation, refunds,
compliance accruals, and intelligent dunning. Version 2.0.

## Feature List

### Core (v1.0)
- Premium schedule creation with auto-generated instalments (annual / semi-annual / quarterly / monthly)
- Full and partial premium collection against instalments
- Debit order / direct-debit mandate management
- Policy-level refund processing with available-balance guard
- Period reconciliation (collections vs refunds)
- Gross-premium calculator with loadings and discounts
- Billing summary analytics

### Enhanced (v2.0)
- **Partial payment & carry-forward** — accumulates sub-full payments; overpayment credited to next instalment automatically
- **Grace-period & lapse state machine** — IRA Kenya / FSCA-compliant transition: `pending → overdue → in_grace → lapsed`
- **Regulatory levy calculator** — itemised IRA Training Levy (0.2 %), PHCF (0.25 %), Stamp Duty per versioned gazette table
- **Payment bounce handling** — reverses collection, levies configurable bounce fee, re-opens instalment for dunning
- **IFRS 17 earned premium (PAA)** — pro-rata temporis written / earned / unearned split per reporting date
- **Dunning engine** — tiered escalation: REMINDER_1 → REMINDER_2 → FORMAL_NOTICE → LAPSE_WARNING with dispatch list
- **Lapse risk scoring** — 0–1 propensity score from payment-history features (days-late trend, partial-pay freq, method volatility)
- **Real-time collection KPIs** — O(1) dashboard: collection ratio, overdue aging buckets, channel mix percentages
- **Chain-hashed audit export** — SHA-256 chain-hashed event log for tamper-evident regulatory submissions

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/prm/health | Health check |
| GET | /api/insurance/prm/describe | Capability description (v2.0) |
| GET | /api/insurance/prm/schedules | List schedules |
| POST | /api/insurance/prm/schedules | Create schedule |
| GET | /api/insurance/prm/schedules/{id} | Get schedule |
| PATCH | /api/insurance/prm/schedules/{id} | Update schedule |
| DELETE | /api/insurance/prm/schedules/{id} | Cancel schedule |
| POST | /api/insurance/prm/schedules/{id}/lapse | Evaluate lapse status |
| POST | /api/insurance/prm/schedules/{id}/earned | IFRS 17 earned premium |
| POST | /api/insurance/prm/schedules/{id}/lapse-score | Lapse risk score |
| GET | /api/insurance/prm/instalments | List instalments |
| GET | /api/insurance/prm/instalments/{id} | Get instalment |
| GET | /api/insurance/prm/instalments/overdue | Overdue instalments |
| POST | /api/insurance/prm/instalments/{id}/collect | Full collection |
| POST | /api/insurance/prm/instalments/{id}/partial | Partial payment |
| POST | /api/insurance/prm/collections/{id}/bounce | Record bounce |
| GET | /api/insurance/prm/collections | List collections |
| POST | /api/insurance/prm/debit-orders | Setup debit order |
| DELETE | /api/insurance/prm/debit-orders/{id} | Cancel debit order |
| POST | /api/insurance/prm/refunds | Process refund |
| GET | /api/insurance/prm/refunds | List refunds |
| POST | /api/insurance/prm/reconcile | Period reconciliation |
| GET | /api/insurance/prm/reconciliations | List reconciliations |
| POST | /api/insurance/prm/dunning | Run dunning cycle |
| POST | /api/insurance/prm/calculate | Gross premium calculation |
| POST | /api/insurance/prm/levies | Statutory levy calculation |
| GET | /api/insurance/prm/kpis | Real-time collection KPIs |
| GET | /api/insurance/prm/summary | Billing summary |
| GET | /api/insurance/prm/audit | Audit trail |
| GET | /api/insurance/prm/audit/chain | Chain-hashed audit export |

## Quick Usage Examples

### 1. Partial M-Pesa payment with automatic carry-forward

```python
svc = PremiumBillingService(tenant_id="acme_ins")

# Policyholder pays KES 3 000 against a KES 5 000 quarterly instalment
partial = await svc.record_partial_payment(
    tenant_id="acme_ins",
    instalment_id="inst-abc123",
    payment_method="mpesa",
    payment_reference="QHG8NZXV12",
    amount=Decimal("3000"),
    collected_by="agent_007",
)
# instalment status → "partial", paid_so_far = 3000
# Pay the remainder later — on full settlement the overpayment auto-credits the next instalment.
```

### 2. IFRS 17 earned premium for month-end close

```python
accrual = await svc.compute_earned_premium(
    tenant_id="acme_ins",
    schedule_id="sch-xyz789",
    reporting_date="2026-06-30",
)
# Returns: written_premium, earned_premium, unearned_premium_reserve
# → post directly to GL journal entries
print(accrual["unearned_premium_reserve"])  # Decimal string, e.g. "45000.00"
```

### 3. Run dunning cycle and dispatch follow-up actions

```python
result = await svc.run_dunning_cycle(
    tenant_id="acme_ins",
    grace_period_days=7,
)
for action in result["dispatches"]:
    print(action["dunning_level"], action["action_required"], action["policy_id"])
# → "FORMAL_NOTICE  Issue formal written notice via registered post  POL-00123"
```

## Integration Notes

`ins_prm` integrates with the following APG capabilities:

| Capability | Integration point |
|---|---|
| `ins_pol` (Policy Admin) | Subscribes to `policy_lapsed` / `lapse_warning` events from `evaluate_lapse_status()` to suspend coverage |
| `ins_clm` (Claims) | Validates premium currency (no outstanding instalments) before claim authorisation |
| `ins_uwt` (Underwriting) | Provides `score_lapse_risk()` scores to underwriting renewal workflows |
| `fin_led` (Finance Ledger) | Consumes `compute_earned_premium()` output for IFRS 17 journal entries |
| `ntf_sms` / `ntf_email` (Notifications) | Receives dunning dispatch lists from `run_dunning_cycle()` |
| `rpt_reg` (Regulatory Reporting) | Consumes `export_audit_chain()` for IRA/FSCA submissions |
