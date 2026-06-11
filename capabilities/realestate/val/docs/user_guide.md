# Property Valuation

**Capability ID**: `realestate_val` | **Domain**: `realestate` | **Version**: `1.0.0`

## Description

Full-cycle property valuation: comparable sales database, DCF model builder with range-validated discount rates, mass appraisal engine (regression, spatial, hedonic, AI AVM), valuation roll with automatic supersession, revaluation cycle management, Red Book sign-off enforcement with independent valuer validation, and structured challenge workflow requiring counter-evidence.

## Installation

```bash
pip install apg-realestate-val
```

## Provides

- `comparable_sales_analysis`
- `dcf_valuation_engine`
- `mass_appraisal_engine`
- `valuation_roll_management`
- `revaluation_cycle_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/val/dashboard` | `realestate_val:view` | Overview |
| `/realestate/val/valuations` | `realestate_val:valuations` | Valuations |
| `/realestate/val/valuations/<id>` | `realestate_val:valuations` | Valuations |
| `/realestate/val/comparables` | `realestate_val:comparables` | Analysis |
| `/realestate/val/dcf` | `realestate_val:dcf` | Models |
| `/realestate/val/mass-appraisal` | `realestate_val:mass_appraisal` | Models |
| `/realestate/val/roll` | `realestate_val:roll` | Roll |
| `/realestate/val/cycles` | `realestate_val:cycles` | Planning |

## Key Service Methods

### Valuer Management

```python
svc = ValService(tenant_id="t1", actor_id="user1")

# Register a panel valuer
valuer = await svc.register_valuer(ValuerCreate(
    tenant_id="t1", name="Jane Doe", grade=ValuerGrade.rics_registered,
    email="jane@firm.com", is_independent=True, created_by="admin",
))

valuers = await svc.list_valuers("t1", independent_only=True)
```

### Comparable Evidence

```python
# Single add
cmp = await svc.add_comparable(ComparableCreate(
    tenant_id="t1", comparable_type=ComparableType.sale,
    address="12 Main St", transaction_date=date(2025, 3, 1),
    price=Decimal("8500000"), area=Decimal("120"), created_by="user1",
))

# Batch import with deduplication
result = await svc.bulk_import_comparables(records=[...], tenant_id="t1")
# → {"inserted": 42, "skipped_duplicate": 3, "validation_errors": 1, ...}

# Apply adjustment matrix
adj = await svc.apply_comparable_adjustments(
    subject_property_id="p1", comparable_id=cmp.id, tenant_id="t1",
    time_adjustment_pct=2.0, size_adjustment_pct=-1.5, condition_adjustment_pct=3.0,
)
# → {"adjusted_price": 8670000, "reliability_score": 82, "reliability_grade": "high", ...}
```

### Valuation Models

```python
# DCF with explicit cash flows
dcf = await svc.dcf_valuation(
    property_id="p1",
    cash_flows=[{"amount": 500000, "label": f"Year {i}"} for i in range(1, 6)],
    discount_rate=0.07, terminal_yield=0.055, tenant_id="t1",
)
# → {"net_capital_value": 7_842_300, "irr_pct": ..., "cash_flow_schedule": [...], ...}

# Sensitivity grid (sweeps DR and exit yield ±150 bps in 50 bp steps)
grid = await svc.dcf_sensitivity_analysis(
    property_id="p1", cash_flows=[...],
    base_discount_rate=0.07, base_exit_yield=0.055, tenant_id="t1",
)
# → {"grid": [...49 scenarios...], "recommended_range_low": 6.9M, "recommended_range_high": 8.8M}

# Income capitalisation
ic = await svc.income_capitalisation(
    property_id="p1", passing_rent=Decimal("480000"),
    market_yield=Decimal("0.055"), tenant_id="t1",
)
# → {"capital_value": 8727272.73, "method": "income_capitalisation"}

# Equivalent yield (NIY + EY + reversionary in one call)
ey = await svc.calculate_equivalent_yield(
    property_id="p1", passing_rent=Decimal("450000"),
    market_rent=Decimal("500000"), purchase_price=Decimal("8000000"),
    unexpired_term_years=4.5, tenant_id="t1",
)
# → {"net_initial_yield_pct": 5.625, "equivalent_yield_pct": 5.94, "reversionary_yield_pct": 6.25}

# Residual land value
rlv = await svc.residual_land_valuation(
    property_id="site1", gross_development_value=Decimal("50000000"),
    build_cost=Decimal("30000000"), tenant_id="t1",
)
# → {"residual_land_value": 4_560_000, "viable": True, "residual_as_pct_gdv": 9.12}

# AVM with confidence bands
avm = await svc.run_avm(
    property_id="p1",
    subject_attributes={"floor_area_sqm": 120, "bedrooms": 3, "condition": 4},
    tenant_id="t1", radius_km=1.5,
)
# → {"value_low": 7.8M, "value_central": 8.4M, "value_high": 9.1M, "confidence": "high"}
```

### Rent Review Modelling

```python
review = await svc.model_rent_review(
    property_id="p1", lease_id="l1",
    passing_rent=Decimal("450000"), open_market_rent=Decimal("510000"),
    tenant_id="t1", review_clause="upward_only",
)
# → {"revised_rent": 510000, "uplift_pct": 13.33, "ifrs16_trigger": True,
#    "next_review_date": "2030-06-11"}
```

### Portfolio Analytics

```python
# Detect properties needing revaluation
triggers = await svc.detect_revaluation_triggers(
    tenant_id="t1", max_age_months=12, ifrs_reporting_window_days=30,
)
# → {"properties_triggered": 7, "triggers": [...sorted by urgency_score...]}

# IAS 40 / IFRS 13 movement report
variance = await svc.portfolio_variance_report(
    tenant_id="t1", current_period="2025-12",
    prior_period_values={"p1": 8000000.0, "p2": 15000000.0},
)
# → {"total_revaluation_surplus_deficit": 950000, "like_for_like_growth_pct": 5.94,
#    "acquisitions": 2, "disposals": 0, "movements": [...]}
```

### Valuation Lifecycle

```python
# Instruct → submit → sign-off → publish
val = await svc.instruct_valuation(ValuationCreate(...))
val = await svc.submit_valuation(val.id, value=Decimal("8500000"), methodology="dcf",
                                  report="RPT-001", tenant_id="t1")
val = await svc.sign_off_valuation(val.id, "t1", "jane_valuer_id", ValuerGrade.rics_registered)
val = await svc.publish_valuation(val.id, "t1")

# Revaluation cycle for entire portfolio
cycle = await svc.revaluation_cycle(
    portfolio_id="port1", effective_date=date(2025, 12, 31), tenant_id="t1",
)
# → {"valuations_instructed": 47, "status": "in_progress"}
```

_(See `service.py` for full signatures and all parameters.)_

## Interoperability

`realestate_val` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use realestate_val;
```

Key integration points:

| Upstream capability | Integration point |
|---------------------|------------------|
| `realestate_ren` | DCF rental income inputs sourced from rent roll |
| `realestate_lea` | IFRS 16 commencement triggers `instruct_valuation()`; rent review triggers `model_rent_review()` |
| `realestate_prm` | Published valuation figure written to property `current_valuation` field |
| `realestate_ins` | Reinstatement cost assessment feeds insurance sum insured |
| `schd` | `detect_revaluation_triggers()` output consumed to schedule revaluation instructions |
| `auth` | Sign-off and challenge review authority gating |
| `audl` | Immutable valuation audit trail |
| `ntfy` | Revaluation due and challenge alerts |

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `REALESTATE_VAL_`.

| Key | Default | Description |
|-----|---------|-------------|
| `dcf.min_discount_rate` | 0.03 | Minimum discount rate |
| `dcf.max_discount_rate` | 0.30 | Maximum discount rate |
| `mass_appraisal.calibration_required` | true | Mandate model calibration before run |
| `valuers.independence_required_for_red_book` | true | Red Book requires independent valuer |
| `revaluation.max_age_months` | 12 | Trigger revaluation after this age |
| `avm.min_comparables` | 3 | Minimum verified comparables for AVM |

## Further Reading

- `service.py` — Business logic (42 async methods)
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and schemas
- `README.md` — Quick reference and API route table
- `WORLD_CLASS_IMPROVEMENTS.md` — Prioritised improvement roadmap (15 items)
