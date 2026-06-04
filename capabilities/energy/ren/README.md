# Renewable Energy

## Overview
Renewable Energy manages the full lifecycle of renewable generation assets — solar PV, wind, hydro, biomass, geothermal and others. It tracks curtailment events with revenue loss accounting, issues and retires Renewable Energy Certificates (RECs) with double-issuance prevention, manages carbon credits requiring third-party verification, administers feed-in tariffs, publishes multi-horizon generation forecasts, and computes performance metrics against benchmarks.

## Capability ID
`energy_ren`

## Provides
| Service | Description |
|---|---|
| `renewable_asset_registry` | Register and manage renewable assets by type, capacity and location |
| `curtailment_tracking` | Record curtailment events with reason, MWh and revenue loss |
| `rec_certificate_management` | Issue, transfer and retire RECs with registry and vintage validation |
| `carbon_credit_management` | Issue and retire carbon credits with verification requirement |
| `feed_in_tariff_management` | Create and activate feed-in tariffs per asset with approval |
| `generation_forecasting` | Publish and version short- to medium-horizon generation forecasts |
| `renewable_performance_analytics` | Track capacity factor, PR ratio, yield vs benchmark |
| `green_energy_reporting` | Portfolio-level renewable generation and certificate reports |

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

## Edge Cases Handled
- REC double-issuance check spans same asset + vintage year + REC type triple
- Retired RECs are permanently immutable — cancel endpoint denied on retired status
- Carbon credits require third-party verification reference on issuance
- FIT requires both type validation and approval in a single create-and-activate call
- Curtailment approval separate from recording to enforce four-eyes principle
- Performance metric stores benchmark deviation for instant deviation alerting

## Composability Notes
- REC certificates feed `energy_bil` for green tariff credits and net metering billing
- Carbon credits feed carbon market settlement in `energy_grd`
- Curtailment events integrate with `energy_grd` for grid congestion analysis
- Generation forecasts feed `energy_grd` unit commitment and economic dispatch
- Asset performance feeds `intel` analytics for portfolio benchmarking
