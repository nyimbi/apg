# Ore Processing & Metallurgy — User Guide

**Capability ID**: `mining_ore` | **Domain**: `mining` | **Version**: `1.1.0`
**© 2025 Datacraft** | nyimbi@gmail.com | www.datacraft.co.ke

---

## Description

Manages ore processing plant operations across the full metallurgical workflow: plant feed tracking, grind circuit control, reagent management with SPC, metallurgical mass balance with closure verification, CIL carbon management, product quality assurance, tailings thickener performance, site water balance, Net Smelter Return computation, ore type classification, and shift/monthly reporting.

---

## Installation

```bash
pip install apg-mining-ore
```

---

## Quick Start

```python
from apg_mining_ore.service import OreService

svc = OreService(tenant_id="my_mine")

# Record plant feed
feed = await svc.plant_feed_record(
    period="2026-05",
    source_blend={"ROM_stockpile": 0.7, "high_grade_ore": 0.3},
    tonnes=45000,
    grade=3.2,   # g/t Au
    moisture_pct=6.5,
    particle_size_p80_mm=180,
)

# Submit and close a metallurgical balance
balance = await svc.metallurgical_balance(
    period="2026-05",
    feed_tonnes=44100,
    concentrate_tonnes=210,
    recovery_pct=91.3,
    tail_grade=0.08,
    feed_grade=3.2,
    concentrate_grade=620.0,
    commodity="Au",
)
closure = await svc.close_metallurgical_balance(
    balance_id=balance["id"],
    feed_dry_t=44100,
    feed_grade=3.2,
    concentrate_dry_t=210,
    concentrate_grade=620.0,
    tailings_dry_t=43890,
    tailings_grade=0.08,
)
print(closure["closure_ok"])          # True
print(closure["recovery_assay_pct"]) # ~91.x
```

---

## Core Workflows

### 1. Plant Feed Recording

Record every processing period's feed tonnage, grade, and source blend:

```python
feed = await svc.plant_feed_record(
    period="2026-05",
    source_blend={"ROM_stockpile": 0.6, "high_grade_ore": 0.4},
    tonnes=50000,
    grade=2.8,
    moisture_pct=7.0,
    particle_size_p80_mm=200,
    feed_type="ROM",
    recorded_by="metallurgist@mine.com",
)
```

`source_blend` fractions must sum to 1.0 (±0.02 tolerance; a warning is logged if outside this range).

---

### 2. Grind Circuit Optimisation

Get PID-style advisory setpoints to hit target P80:

```python
grind = await svc.grind_optimisation_cycle(
    circuit_id="SAG-001",
    current_p80_um=95.0,   # PSA reading
    target_p80_um=75.0,
    mill_speed_pct=74.0,
    water_addition_m3h=180.0,
    ore_hardness_bwi=13.5,
    feed_rate_tph=420.0,
)
print(grind["action"])                        # "grind_finer"
print(grind["recommended_mill_speed_pct"])   # adjusted speed
print(grind["estimated_specific_energy_kwh_t"])
```

P80 deviations > 10% trigger a warning log. Recommended mill speed is clamped to [60%, 85%] of critical speed.

---

### 3. Metallurgical Balance — Closure Verification

The two-product formula is applied to verify mass and metal closure:

```python
result = await svc.close_metallurgical_balance(
    balance_id="<uuid>",
    feed_dry_t=44100,
    feed_grade=3.2,            # g/t Au
    concentrate_dry_t=210,
    concentrate_grade=620.0,   # g/t Au
    tailings_dry_t=43890,
    tailings_grade=0.08,
    tolerance_pct=3.0,         # default
)
# result["closure_ok"] == True|False
# result["mass_closure_error_pct"]   — should be < 3%
# result["metal_closure_error_pct"]  — should be < 3%
# result["recovery_assay_pct"]       — assay-derived recovery
```

If closure fails, the balance record is annotated but not blocked (configure `approval_required=True` to gate publication on closure).

---

### 4. Ore Type Classification

Feed XRF data to assign a geometallurgical domain:

```python
classification = await svc.classify_ore_type(
    source_block_id="BLK-2026-0412",
    xrf_assay={"Au_g_t": 3.5, "As_ppm": 1500, "S_pct": 2.5, "Cu_ppm": 200},
    depth_m=180.0,
    visual_description="pyrite-arsenopyrite veining, dark grey",
)
print(classification["ore_domain"])                    # "refractory"
print(classification["recommended_processing_route"]) # "BIOX_or_POX_then_CIL"
print(classification["expected_recovery_min_pct"])    # 50
```

Domains: `oxide` | `transition` | `primary_sulphide` | `refractory`

---

### 5. Reagent SPC Control

Apply Shewhart control charts to reagent dosage data:

```python
spc = await svc.spc_reagent_control(
    circuit_id="CIL-001",
    reagent_type="cyanide",
    dosage_series=[310, 318, 305, 340, 298, 325, 312, 330, 295, 320],  # g/t
    recovery_series=[91.2, 91.8, 90.5, 92.1, 90.0, 91.5, 91.0, 91.9, 89.8, 91.3],
    target_dosage_g_t=315.0,
)
print(spc["western_electric_violations"])   # indices of out-of-control points
print(spc["dosage_recovery_correlation"])   # Pearson r
print(spc["recommendation"])
```

Requires minimum 5 observations. Returns UCL, LCL, mean, std, drift, and a plain-language recommendation.

---

### 6. CIL Carbon Loading Profile

Track carbon loading across all CIL tanks and detect gradient inversions:

```python
cil = await svc.record_cil_loading(
    circuit_id="CIL-001",
    period="2026-05",
    tank_profiles=[
        {"tank_no": 1, "loaded_carbon_g_t": 7200, "carbon_mass_t": 8.0},
        {"tank_no": 2, "loaded_carbon_g_t": 6100, "carbon_mass_t": 8.0},
        {"tank_no": 3, "loaded_carbon_g_t": 4800, "carbon_mass_t": 8.0},
        {"tank_no": 4, "loaded_carbon_g_t": 3200, "carbon_mass_t": 8.0},
        {"tank_no": 5, "loaded_carbon_g_t": 1800, "carbon_mass_t": 8.0},
        {"tank_no": 6, "loaded_carbon_g_t":  900, "carbon_mass_t": 8.0},
    ],
    solution_grade_mg_l=0.08,
    carbon_inventory_t=48.0,
    elution_due=False,
)
print(cil["loading_gradient_ok"])     # True — decreasing from tank 1 to 6
print(cil["overloaded_tanks"])        # [] — all below 8000 g/t
print(cil["total_gold_locked_kg"])    # gold in carbon inventory
```

---

### 7. Tailings Thickener Performance

```python
thickener = await svc.record_thickener_performance(
    thickener_id="TH-001",
    period="2026-05",
    underflow_solids_pct=52.0,
    overflow_turbidity_ntu=35.0,
    flocculant_dosage_g_t=18.5,
    feed_rate_tph=380.0,
    thickener_area_m2=1590.0,   # π × r² for 45m diameter thickener
    target_underflow_solids_pct=55.0,
    turbidity_limit_ntu=50.0,
)
print(thickener["underflow_on_spec"])        # False — 52% < 55% target
print(thickener["unit_area_loading_t_m2_d"]) # t/m²·day
```

---

### 8. Water Balance and Permit Compliance

```python
water = await svc.record_water_balance(
    period="2026-05",
    fresh_water_intake_m3=85000,
    process_water_recycled_m3=320000,
    tailings_dam_return_m3=45000,
    evaporation_loss_m3=12000,
    effluent_discharged_m3=8000,
    recycled_water_quality={"pH": 8.1, "TSS_mg_l": 42, "CN_mg_l": 0.04},
    permit_limits={"pH": 9.0, "TSS_mg_l": 50, "CN_mg_l": 0.05},
)
print(water["recycle_rate_pct"])      # %
print(water["water_intensity_m3_t"]) # m³ per tonne milled
print(water["permit_compliant"])      # True
```

Each exceedance is individually logged and returned in `compliance_exceedances`.

---

### 9. Net Smelter Return

```python
nsr = await svc.compute_nsr(
    concentrate_grade_g_t=620.0,
    concentrate_tonnes=210.0,
    spot_price_usd_oz=2450.0,
    treatment_charge_usd_t=85.0,
    refining_charge_usd_oz=0.50,
    payability_pct=99.5,
    transport_usd_t=12.0,
    penalty_elements={"As_ppm": 1500},
    commodity="Au",
)
print(nsr["nsr_usd_per_t_concentrate"])  # USD/t
print(nsr["nsr_total_usd"])              # total parcel value
```

Negative NSR triggers a warning log (does not block recording).

---

### 10. Ore Hardness (Bond Work Index)

```python
hardness = await svc.record_ore_hardness(
    source_block_id="BLK-2026-0412",
    bwi_kwh_t=16.8,
    abrasion_index=0.38,
    test_method="Bond_ball_mill",
    ore_type="primary_sulphide",
    sample_depth_m=185.0,
)
print(hardness["hardness_class"])               # "hard"
print(hardness["relative_throughput_factor"])   # 12/16.8 = 0.714 → 28.6% below nameplate
```

BWI > 130% of design (12 kWh/t default) triggers a throughput warning.

---

### 11. Shift Metallurgical Report

```python
from datetime import datetime

report = await svc.generate_shift_met_report(
    shift_start=datetime(2026, 5, 15, 6, 0),
    shift_end=datetime(2026, 5, 15, 14, 0),
    shift_supervisor="J. Kamau",
    shift_label="day",
)
print(report["total_feed_tonnes"])
print(report["critical_deviations"])
print(report["recovery_alert_threshold_pct"])  # mean − 2σ threshold
```

---

## Deviation Alert Lifecycle

```
raise_deviation_alert()
        ↓
acknowledge_deviation()   # operator confirms awareness
        ↓
resolve_deviation()       # root cause documented in resolution_notes
```

List open critical alerts:

```python
alerts = await svc.list_deviation_alerts(open_only=True, alert_level="critical")
```

---

## Reagent Inventory

```python
# Receive delivery
await svc.add_reagent_stock("cyanide", quantity_kg=5000)

# Check levels
inventory = await svc.get_reagent_inventory()
# {"cyanide": 4850.0, "lime": 12400.0, ...}
```

Inventory is automatically decremented on each `record_reagent_usage()` call. Clamped to zero — no negative inventory. Warning logged at < 500 kg.

---

## Process KPI Summary

```python
kpis = await svc.get_process_kpi_summary()
# {
#   "average_recovery_pct": 91.2,
#   "open_deviation_alerts": 3,
#   "critical_deviation_alerts": 1,
#   "total_feed_records": 12,
#   ...
# }
```

---

## Configuration Reference

All keys are tenant-scoped. Set via the `conf` capability or env vars prefixed `MINING_ORE_`.

| Key | Default | Description |
|-----|---------|-------------|
| `plant_feed.feed_grade_required` | `true` | Grade mandatory on all feed records |
| `reagents.cyanide_code_compliance_required` | `true` | ICMC compliance check on cyanide usage |
| `reagents.inventory_tracking_required` | `true` | Maintain per-reagent inventory balance |
| `reagents.low_stock_threshold_kg` | `500` | Warning threshold (kg) |
| `metallurgical_balance.approval_required` | `true` | Approval required before publication |
| `metallurgical_balance.closure_tolerance_pct` | `3.0` | Max allowable mass/metal closure error |
| `cil.safe_loading_limit_g_t` | `8000` | Carbon loading warning threshold |
| `grind.mill_speed_min_pct` | `60` | Lower bound for recommended mill speed |
| `grind.mill_speed_max_pct` | `85` | Upper bound for recommended mill speed |
| `product_quality.specification_check_required` | `true` | Spec check mandatory on quality records |
| `nsr.warn_on_negative` | `true` | Warn when NSR drops below zero |

---

## Interoperability

```apg
use mining_ore;
```

| Upstream | Data Supplied |
|----------|--------------|
| `mining_pro` | Stockpile movements → plant feed source |
| `mining_exp` | Block model grades, ore domain → ore classification |

| Downstream | Data Consumed |
|-----------|--------------|
| `mining_saf` | Off-spec product non-conformance events |
| `mining_env` | Water balance, tailings, ESG Scope 1/2 |
| `scm` | Reagent reorder triggers |
| `fin` | NSR, revenue forecasting |
| `ntfy` | Shift reports, alerts, low-stock warnings |

---

## Further Reading

- `service.py` — Business logic (all async methods with full docstrings)
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoint definitions
- `views.py` — Flask-AppBuilder views and schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 domain-expert improvement proposals
- `SPECIFICATION.md` — Full capability specification
- `tests/` — Unit and integration test suite
