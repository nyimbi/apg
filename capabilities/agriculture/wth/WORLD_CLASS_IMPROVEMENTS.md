# Weather & Climate Analytics — World-Class Improvements

Fifteen improvements that elevate agr_wth from a basic forecast ingestion service to a
full agri-climate intelligence platform competitive with best-in-class precision-ag vendors.

---

### I1. Evapotranspiration (ET₀) Computation
**Category**: Feature
**Justification**: Irrigation scheduling without ET₀ is guesswork. Every modern precision-ag platform (John Deere Operations Center, CropX, Hortau) derives irrigation triggers from Penman-Monteith ET₀. Adding this turns raw forecast data into actionable water-budget numbers, directly reducing over-irrigation costs by 15-30%.
**Implementation**: Implement FAO-56 Penman-Monteith formula using forecast temperature, humidity, wind speed, and solar radiation; expose `compute_et0` that accepts a forecast_id and returns daily ET₀ in mm/day.
**Competitive reference**: John Deere Operations Center (ET-based irrigation recommendations)

---

### I2. Growing Degree Days (GDD) Accumulation Tracking
**Category**: Feature
**Justification**: GDD is the universal crop-phenology clock. Granular GDD tracking is the single feature that separates advisory platforms (Climate Corporation, Corteva Agriscience) from simple weather dashboards. Farmers use it to predict pollination, pest emergence, and harvest timing with day-level precision.
**Implementation**: Accumulate GDD per region/crop using configurable base temperatures; store daily increments derived from forecast records and expose `get_gdd_accumulation(region, crop_type, season_start)`.
**Competitive reference**: Climate Corporation FieldView (GDD tracking by field)

---

### I3. Probabilistic Forecast Ensemble Support
**Category**: Feature
**Justification**: Single-point forecasts mislead; ensemble percentiles (P10/P50/P90) are the industry standard for risk-hedged decisions. Agromonitoring and IBM Environmental Intelligence Suite expose percentile rainfall envelopes. Storing and querying ensembles lets downstream risk models capture uncertainty rather than false precision.
**Implementation**: Extend the forecast schema with `ensemble_members: list[dict]` and a `percentile_summary` field; add `get_forecast_percentiles(region, parameter, date_range)` that returns P10/P25/P50/P75/P90 across stored members.
**Competitive reference**: IBM Environmental Intelligence Suite (ensemble probabilistic forecasts)

---

### I4. Satellite-Derived Vegetation Stress Index (NDVI / VCI)
**Category**: Integration
**Justification**: Weather alone does not reveal actual crop stress — NDVI and the Vegetation Condition Index (VCI) do. Platforms like Descartes Labs and EarthDaily Analytics correlate satellite imagery with climate data to produce early warning 2-4 weeks before yield loss becomes visible. This integration closes the gap.
**Implementation**: Accept NDVI/VCI observations via `ingest_vegetation_index` (region, date, ndvi, vci) and cross-correlate with forecast anomalies in `assess_climate_risk` to weight drought score.
**Competitive reference**: Descartes Labs (NDVI + weather anomaly fusion)

---

### I5. Anomaly Detection on Historical Baselines (Z-score)
**Category**: AI/ML
**Justification**: A 60 mm rainfall event is normal in the Congo basin but extreme in the Sahel. Alert thresholds set in absolute terms produce massive false-positive rates across diverse agroclimatic zones. Z-score anomaly detection against historical normals is how NOAA/CPC and Aclima produce context-aware alerts that operators actually trust.
**Implementation**: After computing monthly normals, calculate Z-score for each incoming forecast parameter; flag observations beyond ±2σ as anomalies in `detect_forecast_anomalies(forecast_id)`, returning ranked anomaly records.
**Competitive reference**: NOAA Climate Prediction Center (standardised anomaly alerts)

---

### I6. Crop-Specific Heat-Unit Windows (Phenological Calendar)
**Category**: Feature
**Justification**: Heat stress thresholds differ dramatically between maize (>35 °C at silking), wheat (>34 °C at anthesis), and coffee (>30 °C). Hard-coding 38 °C for all crops (as the current service does) generates both false positives and misses. Crop-parameterised windows are table-stakes for any platform selling to commodity-specific markets.
**Implementation**: Maintain a `_crop_params` registry keyed by crop_type with base_temp_c, heat_stress_c, frost_kill_c, optimal_rain_mm_season; use these in `assess_climate_risk` and `compute_et0`.
**Competitive reference**: Corteva Agriscience Encirca (crop-specific stress models)

---

### I7. Multi-Source Forecast Consensus Scoring
**Category**: AI/ML
**Justification**: Forecast skill varies by provider, region, and lead time. Blindly accepting any source creates noisy alerts. ECMWF, GFS, and regional NWPs disagree most where uncertainty is highest. Consensus scoring (bias-corrected ensemble mean weighted by historical RMSE) is the mechanism used by The Weather Company Premium for ag customers.
**Implementation**: Add `compute_forecast_consensus(region, valid_date)` that retrieves all sources for the same date, computes mean/spread per parameter, and stores a `consensus` forecast record tagged `source="consensus"`.
**Competitive reference**: The Weather Company Premium Ag (multi-model consensus)

---

### I8. Seasonal Outlook Integration (3-6 Month Probabilistic)
**Category**: Integration
**Justification**: Planting decisions are made 3-6 months ahead, not 10 days. Platforms limited to short-range forecasts cannot inform input purchasing, insurance, or financing decisions. Seasonal outlooks from IRI/CPC or ECMWF seasonal are now standard on platforms sold to commodity trading desks and agri-lenders.
**Implementation**: Add `create_seasonal_outlook(region, season, source, above_normal_pct, near_normal_pct, below_normal_pct)` for tercile probability storage and `get_seasonal_outlook_summary(region, season)` for retrieval and interpretation.
**Competitive reference**: IRI/CPC seasonal outlooks (used by Cargill AgHorizons)

---

### I9. Weather-Triggered Advisory Generation
**Category**: UX
**Justification**: Farmers do not read threshold tables; they need plain-language advisories that say "apply fungicide within 48 hours — leaf wetness window forecast." This is the core differentiator of DTN Progressive Farmer and Bayer Digital Farming (xarvio). Coupling alerts to structured advisory templates converts weather data into decisions.
**Implementation**: Add `generate_weather_advisory(alert_id)` that maps alert severity + parameter to a template registry, returning a structured advisory with recommended action, urgency window (hours), and relevant crop operations.
**Competitive reference**: Bayer xarvio FIELD MANAGER (weather-triggered spray advisories)

---

### I10. Microclimate Zone Interpolation
**Category**: Feature
**Justification**: Regions as defined by administrative boundaries span dozens of kilometres. Elevation, aspect, and proximity to water bodies create microclimates that differ by 3-5 °C and 20-40 % rainfall from the regional mean. CropX and Hortau deploy in-field sensors but spatial interpolation from public weather stations already provides 1 km resolution, which is sufficient for most planting decisions.
**Implementation**: Add `interpolate_microclimate(lat, lon, forecast_id)` using inverse-distance weighting from nearby station records stored in `_history`; return a microclimate-adjusted forecast dict with `interpolation_method` and `station_count` metadata.
**Competitive reference**: CropX (microclimate-adjusted irrigation recommendations)

---

### I11. Carbon Credit Weather Verification
**Category**: Compliance
**Justification**: Voluntary carbon markets (Verra, Gold Standard) require independent weather data verification for soil carbon and biochar methodologies. Agri-carbon platforms (Indigo Ag, Agreena) are gated on providing auditable climate records. Building this as a first-class export differentiates agr_wth for carbon-revenue pathways that represent a $50/ha upside for participating farmers.
**Implementation**: Add `export_carbon_verification_report(region, start_date, end_date)` that produces a signed, timestamped summary of precipitation and temperature records with data provenance, suitable for MRV (measurement, reporting, verification) submissions.
**Competitive reference**: Indigo Ag Carbon (auditable climate record for MRV)

---

### I12. Weather Index Insurance Parametric Trigger Export
**Category**: Compliance
**Justification**: Parametric crop insurance (APA Insurance, Pula Advisors) requires independently computed trigger indices (e.g., cumulative rainfall below 40 mm in 30 days → payout). Integrating trigger computation removes the need for a separate actuarial data feed, enabling agr_wth to be the authoritative settlement source for embedded insurance products.
**Implementation**: Add `compute_insurance_trigger(region, index_type, start_date, end_date, trigger_threshold)` that evaluates the index against stored historical and forecast data, returns `{triggered: bool, index_value, threshold, payout_factor}`.
**Competitive reference**: Pula Advisors (parametric index crop insurance)

---

### I13. Forecast Accuracy Backtesting
**Category**: AI/ML
**Justification**: Without tracking forecast RMSE against eventual observations, operators have no basis for calibrating trust in any provider. This is table-stakes in aviation (TAF verification) and is increasingly demanded by agri-lenders who use weather forecasts to underwrite short-term crop loans.
**Implementation**: Add `record_observation(region, obs_date, parameter, observed_value)` and `compute_forecast_accuracy(region, source, parameter, date_range)` that returns MAE, RMSE, and bias for matched forecast/observation pairs.
**Competitive reference**: DTN (forecast verification scores published per-region)

---

### I14. Real-Time Push Notification Dispatch
**Category**: UX
**Justification**: Email-only alert delivery has a median open-time of 6.4 hours — unacceptable for a frost warning requiring action within 2 hours. Platforms like aWhere and Taranis deliver SMS/push notifications within minutes of threshold breach. Adding a notification dispatch step to `_evaluate_thresholds` closes this gap without requiring a separate notification service.
**Implementation**: Add a `_notification_handlers: list[Callable]` registry; `register_notification_handler(handler)` allows callers (SMS gateway, FCM push, Telegram bot) to subscribe; `_evaluate_thresholds` calls handlers for each triggered alert with severity-based priority tagging.
**Competitive reference**: Taranis (real-time push alerts for field agronomists)

---

### I15. Water Stress Index (WSI) Time-Series
**Category**: Feature
**Justification**: The FAO Water Stress Index — actual ET divided by potential ET — is the most direct quantitative signal for irrigation need and drought-induced yield loss. It is the primary variable in FAO's AquaCrop model and is used by irrometer and Netafim to schedule drip irrigation. Tracking WSI across a season gives lenders and insurers a continuous proxy for crop water status without requiring in-field sensors.
**Implementation**: Add `compute_water_stress_index(region, date_range)` that derives actual ET from rainfall and soil water-holding defaults, divides by Penman-Monteith ET₀ (from I1), and returns a daily WSI time-series with drought classification bands.
**Competitive reference**: Netafim Digital Farming (WSI-driven drip scheduling)
