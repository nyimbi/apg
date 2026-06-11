# Lease Management — User Guide

**Capability**: `realestate_lea`
**Version**: 2.0
**Copyright**: © 2025 Datacraft

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Lease Lifecycle](#lease-lifecycle)
4. [IFRS 16 / ASC 842 Accounting](#ifrs-16--asc-842-accounting)
5. [Rent Management](#rent-management)
6. [Options and Incentives](#options-and-incentives)
7. [Portfolio Analytics and Reporting](#portfolio-analytics-and-reporting)
8. [Advanced Analytics (v2)](#advanced-analytics-v2)
9. [Configuration Reference](#configuration-reference)
10. [Error Reference](#error-reference)

---

## Overview

`realestate_lea` manages the complete lease lifecycle for commercial, retail, industrial, office, residential, and ground leases. It enforces IFRS 16 and ASC 842 accounting rules, tracks rent escalations and reviews, monitors option windows, and provides portfolio-level analytics including WALT, maturity profiles, and KPI dashboards.

All service methods are `async`. Inject a `LeaseManagementService` instance at application startup. In production, pass a SQLAlchemy async session via `db_session`; for testing, the default in-memory dict store is used.

---

## Quick Start

```python
from capabilities.realestate.lea.service import LeaseManagementService

svc = LeaseManagementService(tenant_id="t_acme", actor_id="user_001")

# 1. Create a draft lease
lease = await svc.create_lease(
    property_id="prop_001",
    tenant_id="t_acme",
    lease_type="office",
    start_date="2025-01-01",
    end_date="2030-12-31",
    rent=150000,        # monthly
    currency="KES",
    payment_frequency="monthly",
    options={"discount_rate": "0.07", "floor_area_sqm": 800},
)

# 2. Execute (sign) the lease
await svc.execute_lease(lease["id"], executed_by="user_001", execution_date="2024-12-01")

# 3. Classify and compute IFRS 16
await svc.classify_lease_ifrs16(lease["id"])
await svc.calculate_lease_liability(lease["id"])
await svc.calculate_rou_asset(lease["id"])
```

---

## Lease Lifecycle

### States

```
heads_of_terms → draft → active → [holding_over] → terminated/surrendered/renewed
```

### Creating a Lease

Use `create_lease()` for a quick positional-argument call or `create_lease_v2(LeaseCreate)` for the validated Pydantic v2 entrypoint. `create_lease_v2` is preferred in production as it runs domain rule assertions before storing the record.

**Required options for IFRS 16**: include `discount_rate` (or `implicit_rate` / `ibr`) in the `options` dict. The floor area (`floor_area_sqm`) is required for cost-per-sqm analytics.

### Executing a Lease

`execute_lease(lease_id, executed_by, execution_date)` transitions the lease from `draft` to `active`. The lease must have `abstraction_verified=True` (or set `options.skip_abstraction_check=True` for testing only).

### Amending a Lease

`amend_lease(lease_id, amendment_type, new_terms, effective_date, reason)` applies a formal amendment. Amendment types: `rent_change`, `extension`, `space_change`, `assignment`, `sublease`. Any amendment that modifies the financial terms sets `requires_ifrs16_remeasurement=True` — follow up with `lease_modification_remeasurement()`.

### Renewing a Lease

`renew_lease(lease_id, new_terms, renewal_date)` marks the original lease `renewed`, creates a successor lease with the new terms, and links them via `predecessor_lease_id` / `successor_lease_id`.

### Surrendering or Terminating

- `surrender_lease(lease_id, surrender_date, agreed_compensation)` — records agreed compensation and marks the lease `surrendered`.
- `terminate_lease(lease_id, termination_type, effective_date, notice_date)` — termination types: `expiry`, `break_option`, `landlord_notice`, `forfeiture`.

---

## IFRS 16 / ASC 842 Accounting

### Classification

```python
result = await svc.classify_lease_ifrs16(lease_id)
# result["classification"]: "finance" | "operating"
# result["criteria_met"]: list of criteria that triggered finance classification
```

Finance lease indicators (any one triggers finance classification):
- Lease term ≥ 75% of economic life
- PV of payments ≥ 90% of fair value
- Transfer of ownership
- Bargain purchase option
- Specialised asset

Pass relevant flags in `lease.options`: `transfer_of_ownership`, `bargain_purchase_option`, `specialised_asset`, `pv_substantially_all_fair_value`, `economic_life_months`.

### ROU Asset and Lease Liability

Call in this order after classification:

```python
liability = await svc.calculate_lease_liability(lease_id)
rou       = await svc.calculate_rou_asset(lease_id)
```

ROU asset adjustments sourced from `options`:
- `initial_direct_costs`
- `incentives_paid_to_lessor`
- `incentives_received_from_lessor`
- `restoration_costs`

### Full Amortisation Schedule

```python
schedule = await svc.full_amortisation_schedule(lease_id, summarise_by="annual")
# schedule["schedule"]: list of {year, opening_balance, total_payment, total_interest, total_principal, closing_balance}
```

`summarise_by` options: `monthly` (default), `quarterly`, `annual`. The complete table covers every period from commencement to expiry — required for auditor deliverables.

### Period Entries and Payments

```python
# Monthly journal entries
journals = await svc.ifrs16_journal_entries(lease_id, period="2025-06")

# Process a lease payment (splits into interest + principal)
receipt = await svc.process_lease_payment(lease_id, payment_amount=150000, payment_date="2025-06-01")
```

### Modification Remeasurement

Any change to lease term, rent, or discount rate triggers remeasurement under IFRS 16.45:

```python
result = await svc.lease_modification_remeasurement(
    lease_id,
    event_type="revised_payment",   # scope_change | revised_payment | index_change | rate_change | reassessment
    new_terms={"rent": 165000, "discount_rate": "0.075"},
)
```

For CPI-indexed leases, use `apply_cpi_remeasurement(lease_id, current_cpi, actor_id)` which reads `lease.variable_payment_indexed_to_cpi` and `lease.cpi_base_index`.

### IFRS 16 Disclosures

```python
notes = await svc.ifrs16_disclosure_notes(fiscal_year="2025")
```

Returns all IFRS 16.53–59 disclosure items including maturity analysis, weighted average discount rate, and ROU asset carrying amounts by class.

---

## Rent Management

### Generating Demands

```python
demand = await svc.generate_rent_demand(lease_id, period="2025-06")
# Returns amount_due, arrears_brought_forward, total_due, due_date
```

### Receipts and Allocation

```python
receipt = await svc.process_rent_receipt(
    lease_id, amount=150000, payment_date="2025-06-05", payment_method="bank_transfer"
)
# FIFO allocation to oldest unpaid demands
# receipt["allocations"]: per-demand allocation detail
# receipt["unallocated_balance"]: any surplus after clearing all demands
```

### Arrears Analysis

```python
arrears = await svc.calculate_rent_arrears(lease_id, as_of_date="2025-06-30")
# arrears["aged_analysis"]: {"0_30": ..., "31_60": ..., "61_90": ..., "over_90": ...}
```

### Rent Escalation

```python
result = await svc.apply_rent_escalation(
    lease_id,
    escalation_type="fixed_percentage",  # fixed_percentage | CPI_linked | market_review | stepped
    rate=0.05,
    effective_date="2026-01-01",
)
```

For `stepped` escalations, `rate` is the new absolute monthly rent amount.

### Multi-Year Projection

```python
projection = await svc.project_rent_escalation_schedule(lease_id, years=5, cpi_forecast=0.04)
# projection["projection"]: [{year, monthly_rent, annual_rent, escalation_applied, cumulative_increase_pct}]
```

### Service Charge Reconciliation

```python
recon = await svc.service_charge_reconciliation(property_id="prop_001", period="2025")
# Returns budget vs. actual per line item, tenant allocation, and action (invoice/refund/nil)
```

---

## Options and Incentives

### Renewal Option Assessment

```python
assessment = await svc.assess_renewal_option(lease_id, renewal_date="2029-01-01")
# assessment["reasonably_certain"]: bool
# assessment["probability_score"]: 0–1
# assessment["ifrs16_implication"]: instruction to include/exclude renewal period
```

Factors scored: rent below market, leasehold improvements, management intent, relocation cost, remaining term. Set these in `lease.options`: `market_rent`, `leasehold_improvements_value`, `management_intent_to_renew`, `high_relocation_cost`.

### Break Option Assessment

```python
assessment = await svc.assess_termination_option(lease_id, break_date="2027-06-01")
# assessment["reasonably_certain_to_terminate"]: bool
```

### Break Option Cost Model

```python
cost = await svc.model_break_option_cost(lease_id, break_date="2027-06-01")
# cost["components"]: {break_penalty, unamortised_incentives, dilapidations_estimate, fitout_write_off, relocation_cost}
# cost["recommendation"]: "exercise_break" | "stay"
# cost["net_saving_from_break"]: NPV(remaining payments) - total_break_cost
```

Configure components in `lease.options`: `break_penalty`, `dilapidations_estimate`, `fitout_book_value`, `relocation_cost`.

### Rent-Free Periods

```python
rfp = await svc.record_rent_free_period(
    lease_id, free_from="2025-01-01", free_to="2025-06-30", type="initial_rent_free"
)
# rfp["total_value"]: total rent value of the free period
# rfp["ifrs16_note"]: guidance on IFRS 16 treatment
```

---

## Portfolio Analytics and Reporting

### Portfolio Summary

```python
summary = await svc.lease_portfolio_summary(filters={"status": "active", "lease_type": "office"})
# Returns total_annual_rent, total_rou_assets, total_lease_liabilities, by_status, by_type
```

### WALT

```python
walt = await svc.weighted_average_lease_term()
# Returns float (years); weighted by annual rent
```

### Maturity Profile

```python
profile = await svc.lease_maturity_profile(years=5)
# profile["maturity_profile"]: [{year, expiring_lease_count, expiring_annual_rent, pct_of_portfolio}]
```

### Expiry Pipeline

```python
pipeline = await svc.lease_expiry_pipeline(days_ahead=180)
# Sorted by days_remaining; urgency: critical (<30d) | high (<90d) | medium (<180d) | low
```

### Comprehensive Portfolio Analytics

```python
analytics = await svc.portfolio_lease_analytics(tenant_id="t_acme")
# PortfolioLeaseAnalytics Pydantic model with all KPIs, top leases by liability, sublease income
```

---

## Advanced Analytics (v2)

### KPI Dashboard

```python
kpis = await svc.lease_kpi_dashboard(tenant_id="t_acme", period="2025-06")
```

| KPI | Description |
|-----|-------------|
| `occupancy_rate_pct` | Active leases / total leases × 100 |
| `wault_years` | Rent-weighted average unexpired term |
| `rent_collection_efficiency_pct` | Receipts / demands raised × 100 |
| `avg_lease_term_months` | Mean original term of active leases |
| `leases_expiring_90d/180d/365d` | Count by urgency horizon |
| `total_annual_rent` | Sum of active lease annual rents |
| `total_rou_assets` | Sum of ROU asset carrying values |
| `total_lease_liabilities` | Sum of lease liability balances |
| `modifications_total` | Count of modification records |
| `avg_escalation_rate_pct` | Mean rate across applied escalations |

### ERV Benchmarking

```python
erv = await svc.benchmark_against_erv(lease_id, erv_per_sqm_annual=12000.0)
# erv["over_rented"]: bool (passing > ERV)
# erv["reversion_potential"]: annual delta to ERV
# erv["months_to_reversion"]: periods until next rent review
```

`floor_area_sqm` must be set in `lease.options` for per-sqm analysis.

### Discount Rate Sensitivity

```python
sens = await svc.discount_rate_sensitivity(lease_id, rate_min=0.03, rate_max=0.10, rate_step=0.005)
# sens["sensitivity_matrix"]: [{rate_pct, lease_liability, rou_asset, delta_vs_base, delta_per_100bps}]
```

Use to support auditor sensitivity queries and treasury stress tests.

### Holding-Over Detection

```python
holding_over = await svc.detect_holding_over(as_of_date="2025-12-31", holding_over_uplift_pct=0.25)
# Transitions expired active leases to "holding_over" status
# Applies +25% uplift to passing rent (configurable)
# Returns list of transitioned leases with holding_over_rent
```

Run as a scheduled job (monthly recommended) to prevent silent status drift.

### Data Integrity Validation

```python
report = await svc.validate_lease_data_integrity(lease_id)
# report["passed"]: bool
# report["issues"]: [{check, severity, detail}]  — errors that must be fixed
# report["warnings"]: [{check, severity, detail}] — items requiring attention
```

Checks performed:
- Rent review dates within lease term
- Option exercise windows do not extend beyond expiry
- Sublease term within head lease term
- IFRS 16 ROU + liability fields populated together or both absent
- Active lease not past expiry without holding-over transition
- Active lease has positive rent

---

## Configuration Reference

Set in `lease.options` at creation or via `amend_lease`:

| Key | Type | Description |
|-----|------|-------------|
| `discount_rate` | str/float | Annual IBR used for IFRS 16 PV calculations |
| `implicit_rate` | str/float | Lessor's implicit rate (preferred over IBR) |
| `ibr` | str/float | Incremental borrowing rate (fallback) |
| `floor_area_sqm` | float | Lettable floor area for per-sqm analytics |
| `economic_life_months` | int | Asset economic life for IFRS 16 classification (default 240) |
| `escalation_type` | str | `fixed_percentage` / `CPI_linked` / `stepped` / `market_review` |
| `escalation_rate` | float | Annual rate for fixed/CPI escalations |
| `escalation_frequency_months` | int | How often escalation applies (default 12) |
| `market_rent` | float | Current market rent for ERV and option assessments |
| `break_penalty` | float | Financial penalty for exercising break option |
| `dilapidations_estimate` | float | Estimated terminal dilapidation cost |
| `fitout_book_value` | float | Book value of leasehold improvements at break date |
| `relocation_cost` | float | Estimated cost of relocating to alternative premises |
| `management_intent_to_renew` | bool | Management statement on renewal intent |
| `leasehold_improvements_value` | float | Sunk cost in leasehold improvements |
| `initial_direct_costs` | float | Legal fees, commission — added to ROU asset |
| `restoration_costs` | float | Reinstatement obligation — added to ROU asset |
| `variable_payment_indexed_to_cpi` | bool | CPI-linked lease flag for remeasurement |
| `cpi_base_index` | float | CPI index at lease commencement (default 100) |

---

## Error Reference

| Error Pattern | Cause | Fix |
|---------------|-------|-----|
| `rule_denied:activation_requires_verified_abstraction` | Lease activated before abstraction verified | Call `verify_abstraction()` first or set `options.skip_abstraction_check=True` |
| `AssertionError: rent must be positive` | Zero or negative rent passed | Pass a positive rent value |
| `AssertionError: end_date must be after start_date` | Date ordering wrong | Check date strings |
| `AssertionError: lease liability not calculated` | `full_amortisation_schedule` called before `calculate_lease_liability` | Call `calculate_lease_liability` first |
| `AssertionError: lease is not indexed to CPI` | `apply_cpi_remeasurement` on non-CPI lease | Set `options.variable_payment_indexed_to_cpi=True` |
| `ValueError: modification approval failed` | `handle_lease_modification` could not auto-approve | Check `approved_by` is provided and non-null |
| `rule_denied:option_exercise_requires_notice` | Option exercised without serving notice | Set `notice_served=True` after serving notice |
| `rule_denied:option_exercise_window_required` | Exercise attempted outside the window | Check `exercise_from`/`exercise_to` dates |

---

*User Guide v2.0 — last updated 2026-06-11*
