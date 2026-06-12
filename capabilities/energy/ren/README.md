# Renewable Energy

© 2025 Datacraft | Author: Nyimbi Odero

## Overview

`energy_ren` manages the full lifecycle of renewable generation assets — solar PV, wind, hydro, biomass, geothermal and others. It tracks curtailment events with revenue loss accounting, issues and retires Renewable Energy Certificates (RECs) with double-issuance prevention, manages carbon credits requiring third-party verification, administers feed-in tariffs, publishes multi-horizon generation forecasts, and computes performance metrics against benchmarks.

v2.0 adds SCADA streaming ingest, probabilistic forecasting, PPA lifecycle management, VPP coordination, real-time LMP-based curtailment valuation, predictive maintenance scoring, and a bankability due-diligence export — closing the gap against GE Digital APM, Siemens Omnivise, and Stem Athena.

## Capability ID

`energy_ren`

## Provides

| Service | Description |
|---|---|
| `renewable_asset_registry` | Register and manage renewable assets by type, capacity and location |
| `curtailment_tracking` | Record curtailment events with reason, MWh and LMP-based revenue loss |
| `rec_certificate_management` | Issue, transfer and retire RECs with registry, vintage and hourly interval validation |
| `carbon_credit_management` | Issue and retire carbon credits with multi-standard workflow and buffer pool |
| `feed_in_tariff_management` | Create and activate feed-in tariffs; schedule degression and CPI escalation |
| `generation_forecasting` | Publish versioned forecasts with P10/P50/P90 quantile uncertainty bands |
| `renewable_performance_analytics` | IEC 61724-3 normalised PR, capacity factor, P50 benchmarking |
| `green_energy_reporting` | Portfolio-level generation, certificate, and carbon net-position reports |
| `ppa_lifecycle` | PPA contract modelling, settlement, deemed energy, and exposure reporting |
| `vpp_coordination` | Aggregate distributed assets into a VPP; dispatch across member assets |
| `asset_health_monitoring` | Component-level predictive maintenance scoring from SCADA anomalies |
| `market_intelligence` | Real-time LMP ingestion; curtailment revaluation at actual market price |
| `bankability_package` | Due-diligence data-room export for project finance and green bonds |

## Requires

| Capability | Reason |
|---|---|
| `auth` | User authentication and asset permissions |
| `audl` | Immutable trail for REC issuance, transfer and retirement |
| `mten` | Multi-tenant asset data isolation |
| `conf` | Asset and certificate configuration |
| `ntfy` | Curtailment and performance deviation alerts |
| `wflo` | Curtailment approval and FIT activation workflows |
| `moni` | Real-time asset generation monitoring |
| `comp` | Regulatory compliance for RECs and carbon credits |
| `mqeb` | Event streaming for certificate lifecycle |
| `schd` | Scheduled forecast publication and KPI calculation |

## Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `curtailment.mwh_tracking` | bool | true | Track curtailed MWh per event |
| `curtailment.revenue_loss_tracking` | bool | true | Track revenue loss per curtailment |
| `recs.registry_required` | bool | true | Registry name required on issuance |
| `carbon_credits.verification_required` | bool | true | Third-party verification required |
| `forecasting.model_versioning` | bool | true | Version each published forecast model |

## API Routes

| Path | Method | Description | Permission |
|---|---|---|---|
| `/energy-ren/api/v1/dashboard` | GET | Portfolio dashboard | `energy_ren:view` |
| `/energy-ren/api/v1/assets` | GET | List renewable assets | `energy_ren:assets` |
| `/energy-ren/api/v1/assets` | POST | Register asset | `energy_ren:assets` |
| `/energy-ren/api/v1/assets/<id>` | GET | Asset detail | `energy_ren:assets` |
| `/energy-ren/api/v1/curtailment` | POST | Record curtailment | `energy_ren:curtailment` |
| `/energy-ren/api/v1/curtailment/<id>/approve` | PUT | Approve curtailment | `energy_ren:curtailment` |
| `/energy-ren/api/v1/recs` | GET | List RECs | `energy_ren:recs` |
| `/energy-ren/api/v1/recs` | POST | Issue REC | `energy_ren:recs` |
| `/energy-ren/api/v1/recs/<id>/transfer` | PUT | Transfer REC | `energy_ren:recs` |
| `/energy-ren/api/v1/recs/<id>/retire` | PUT | Retire REC (irreversible) | `energy_ren:recs` |
| `/energy-ren/api/v1/carbon-credits` | POST | Issue carbon credit | `energy_ren:carbon_credits` |
| `/energy-ren/api/v1/carbon-credits/<id>/retire` | PUT | Retire carbon credit | `energy_ren:carbon_credits` |
| `/energy-ren/api/v1/feed-in-tariffs` | POST | Create FIT | `energy_ren:feed_in_tariffs` |
| `/energy-ren/api/v1/forecasting` | POST | Publish forecast | `energy_ren:forecasting` |
| `/energy-ren/api/v1/performance` | POST | Record performance metric | `energy_ren:performance` |

## Quick Start

```python
from capabilities.energy.ren.service import RenewableEnergyService

svc = RenewableEnergyService(tenant_id="acme", actor_id="ops-user")

# Register a 50 MW solar farm
asset = svc.register_asset(
    asset_id="sol-001", tenant_id="acme", name="Rift Valley Solar",
    renewable_type="solar_pv", capacity_mw=50.0, owner_id="acme-energy",
    commissioning_date="2024-01-15", location_reference="KE-RIFT-01",
)

# Record hourly generation
gen = await svc.energy_generation_record(
    asset_id="sol-001", timestamp="2025-06-01T08:00:00Z",
    kwh_generated=18500.0, irradiance_w_m2=720.5,
)

# Issue an I-REC for a verified month
rec = await svc.rec_certificate_create(
    asset_id="sol-001", period="2025-06", mwh_generated=12400.0,
    registry="I-REC", rec_type="I-REC",
)

# Calculate carbon credits (CDM/VCS)
credits = await svc.carbon_credit_calculate(
    asset_id="sol-001", period="2025-06",
    baseline_emission_factor_tco2e_mwh=0.82,
)

# Portfolio analytics
report = await svc.renewable_analytics(period="2025-06")
```

## New Methods

### `energy_generation_record` — high-resolution generation ingest

```python
gen = await svc.energy_generation_record(
    asset_id="wind-007",
    timestamp="2025-06-01T14:00:00Z",
    kwh_generated=32100.0,
    wind_speed_m_s=9.4,
    capacity_factor_pct=None,   # auto-computed from nameplate
    availability_pct=98.5,
)
# Returns: id, mwh_generated, capacity_factor_pct, recorded_at
```

### `carbon_credit_calculate` — CDM/VCS methodology

```python
calc = await svc.carbon_credit_calculate(
    asset_id="sol-001",
    period="2025-06",
    baseline_emission_factor_tco2e_mwh=0.82,   # KERC East Africa grid
    leakage_pct=3.0,
    uncertainty_deduction_pct=2.0,
)
# Returns: gross_credits_tco2e, net_credits_tco2e, generation_records_used
```

### `curtailment_log` — async curtailment with revenue tracking

```python
event = await svc.curtailment_log(
    asset_id="sol-001",
    period="2025-06",
    curtailed_mwh=85.0,
    reason="grid_constraint",
    operator_reference="KETRACO-REF-4421",
    revenue_loss=None,    # defaults to $50/MWh; override with LMP in v2
    currency="USD",
)
```

### `renewable_portfolio_standard_compliance` — RPS assessment

```python
rps = await svc.renewable_portfolio_standard_compliance(
    utility_id="kplc-nairobi",
    period="2025-06",
    rps_target_pct=40.0,
    total_sales_mwh=250_000.0,
)
# Returns: actual_rps_pct, compliant, deficit_mwh, rec_credits_available
```

### `green_tariff_offering` — backed green product

```python
product = await svc.green_tariff_offering(
    product_name="RiftValley 100% Renewable",
    eligible_assets=["sol-001", "wind-007"],
    premium=0.012,          # $/kWh above base tariff
    currency="USD",
    min_commitment_months=12,
    renewable_content_pct=100.0,
)
```

## Business Rules

| Rule | Condition | Effect |
|---|---|---|
| `tenant_context_required` | tenant_context_present=False | deny |
| `renewable_type_supported` | type not in supported list | deny |
| `asset_capacity_positive` | capacity_mw <= 0 | deny |
| `curtailment_reason_supported` | reason not in supported list | deny |
| `curtailment_mwh_positive` | curtailed_mwh <= 0 | deny |
| `rec_double_issuance_denied` | same asset+vintage+type already issued | deny |
| `rec_retirement_irreversible` | cancel on already-retired REC | deny |
| `carbon_credit_verification_required` | verification_present=False | deny |
| `feed_in_tariff_approval_required` | activate FIT without approval | deny |
| `cross_tenant_denied` | cross_tenant_access=True | deny |
| `privileged_ren_agent_requires_human_approval` | agent curtailment without human approval | deny |

## Data Models

| Model | Key Fields |
|---|---|
| `RenewableAsset` | id, name, renewable_type, capacity_mw, status, commissioning_date, location_reference |
| `CurtailmentEvent` | id, asset_id, reason, curtailed_mwh, revenue_loss, currency, status |
| `RecCertificate` | id, asset_id, rec_type, quantity_mwh, vintage_year, registry, status |
| `CarbonCredit` | id, asset_id, credit_type, quantity_tco2e, standard, verification_reference, status |
| `FeedInTariff` | id, asset_id, fit_type, rate_per_kwh, currency, effective_date, approved_by |
| `GenerationForecast` | id, asset_id, forecast_type, horizon, values, model_version, rmse, mae |
| `PerformanceMetric` | id, asset_id, metric_type, value, unit, benchmark_value, deviation |

## Streaming Events

- `renewable_asset_registered` / `asset_status_changed`
- `curtailment_event_created` / `curtailment_event_approved`
- `rec_issued` / `rec_transferred` / `rec_retired`
- `carbon_credit_issued` / `carbon_credit_retired`
- `feed_in_tariff_activated`
- `generation_forecast_published`
- `performance_metric_calculated`
- `scada_anomaly_detected` / `maintenance_alert_recommended`
- `vpp_dispatch_issued` / `verification_requested`

## Edge Cases Handled

- REC double-issuance check spans same asset + vintage year + REC type triple
- Retired RECs are permanently immutable — cancel endpoint denied on retired status
- Carbon credits require third-party verification reference on issuance
- FIT requires both type validation and approval in a single create-and-activate call
- Curtailment approval separate from recording to enforce four-eyes principle
- Performance metric stores benchmark deviation for instant deviation alerting
- Generation capped at 105% of nameplate capacity × interval to catch sensor faults

## World-Class Enhancements (v2.0)

1. **Real-Time SCADA Ingestion** — bulk 1-minute telemetry ingest (`voltage`, `frequency`, `active/reactive power`, `inverter temps`, `string PV current`); rolling 15-min capacity factor; 3σ anomaly detection emitting `scada_anomaly_detected`.

2. **Hourly REC Vintage Matching** — extend `RecCertificate` with `interval_start`/`interval_end`; greedy matching algorithm for 24/7 CFE procurement (RE100, Google CFE standard); returns matched pairs, unmatched hours, and 0–100% matching score.

3. **FIT Degression and CPI Escalation** — `FitSchedule` with `[(effective_date, rate, mechanism)]`; `compute_current_fit_rate` walks schedule applying CPI index or degression; `project_fit_revenue` returns NPV of 20–25 year FIT cash flows.

4. **Multi-Standard Carbon Certification Workflow** — `CarbonProject` with monitoring periods, verifier assignment, buffer pool (10–20% VCS), and `draft → submitted → verified → issued` state machine; fires `verification_requested` event.

5. **Probabilistic Generation Forecasting** — P10/P50/P90 quantile values on `GenerationForecast`; `calibrate_forecast_model` fits quantile regression forest on NWP inputs; `evaluate_forecast_skill` computes CRPS, sharpness, reliability.

6. **Predictive Maintenance Scoring** — `AssetHealthEvent` with component, fault code, severity 1–5; `score_asset_health` produces 0–100 score per component from SCADA anomaly history; fires `maintenance_alert_recommended` below tenant threshold.

7. **PPA Lifecycle Management** — `PowerPurchaseAgreement` with fixed/indexed/proxy-revenue-swap contract types, floor/cap price, buyer credit rating; `calculate_ppa_settlement` computes deemed energy for curtailment beyond agreed limits; `ppa_exposure_report` under P10/P50/P90 scenarios.

8. **Grid Interconnection Queue Tracker** — `InterconnectionStudy` with queue position, study phase, ISO region, deposit tracking; `advance_interconnection_phase` enforces deposit-paid precondition; `interconnection_queue_report` by ISO/phase.

9. **Revenue Stack Optimisation** — LP/MILP dispatch over 96 half-hourly periods maximising energy + capacity + ancillary service revenue subject to ramp rate and SOC constraints; returns `DispatchSchedule` with per-interval MW bids and expected revenue.

10. **Virtual Power Plant Coordination** — `VirtualPowerPlant` aggregates distributed assets for wholesale market participation; `register_vpp` validates aggregate capacity meets market minimums; `dispatch_vpp` distributes signal using priority-ranked allocation.

11. **Automated Regulatory Filing** — `RegulatoryFiling` with `draft → submitted → accepted/rejected` workflow; `generate_regulatory_report` maps to EIA-923, Ofgem RO, I-REC MMS schemas; `submit_filing` posts to regulator APIs with audit-trail payload storage.

12. **IEC 61724-3 Weather Normalisation** — `WeatherNormalisation` with POA irradiance, ambient/module temp, soiling loss; `compute_normalised_performance_ratio` separates equipment degradation from weather variance; `benchmark_against_p50` compares normalised PR to EYA P50.

13. **Carbon Portfolio Netting and Marketplace** — `compute_scope2_net_position` aggregates MWh → tCO2e using regional grid EF, nets against consumption; `list_marketplace_offers` for price discovery; `execute_credit_trade` with atomic cross-tenant transfer.

14. **Bankability Due-Diligence Package** — `generate_due_diligence_package` compiles generation vs P50/P90, curtailment root-cause log, REC chain, verification certificates, O&M budget, PPA scorecard; returns `DueDiligencePackage` with cryptographic integrity hash.

15. **Real-Time LMP Curtailment Valuation** — `MarketPrice` model with node-level LMP, congestion, loss, energy components; `fetch_realtime_lmp` queries ISO APIs (CAISO OASIS, ERCOT, PJM) with `BoundedCache`; `revalue_curtailment_at_lmp` replaces flat $50/MWh with actual market price.

## Composability Notes

- REC certificates feed `energy_bil` for green tariff credits and net metering billing
- Carbon credits feed carbon market settlement in `energy_grd`
- Curtailment events integrate with `energy_grd` for grid congestion analysis
- Generation forecasts feed `energy_grd` unit commitment and economic dispatch
- Asset performance feeds `intel` analytics for portfolio benchmarking
- PPA settlements feed `energy_bil` for revenue reconciliation
- VPP dispatch signals originate from `energy_grd` ancillary service dispatch
