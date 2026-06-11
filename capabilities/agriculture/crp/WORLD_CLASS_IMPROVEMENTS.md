# Crop Management (agr_crp) — World-Class Improvements

## Overview

15 targeted improvements to elevate agr_crp from a basic CRUD service to a decision-grade precision agriculture platform that competes with commercial products like John Deere Operations Center, Climate FieldView, and Granular Insights.

---

### I1. Growing Degree Day (GDD) Accumulation Engine
**Category**: AI/ML
**Justification**: GDD is the agronomic gold standard for predicting crop maturity, pest pressure windows, and harvest timing. Without it, planting calendars are guesswork. John Deere Operations Center and Climate FieldView both surface GDD dashboards as primary KPIs. Farmers who can predict days-to-tassel reduce post-harvest losses by 8–15%.
**Implementation**: Integrate temperature time-series (daily Tmax/Tmin) against a crop-specific base temperature; accumulate `(Tmax + Tmin)/2 - T_base` clamped to [0, cap] per day; expose `get_gdd_accumulation(crop_id, as_of)` returning accumulated GDD and predicted days to next phenological stage.
**Competitive reference**: Climate FieldView GDD Tracker, John Deere Operations Center Heat Units

---

### I2. Yield Gap Analysis with Benchmarking
**Category**: AI/ML
**Justification**: Knowing raw yield is worthless without context. Yield gap analysis (actual vs. water-limited potential vs. theoretical maximum) is the framework used by GYGA (Global Yield Gap Atlas) and underpins financing decisions by agricultural lenders. Granular Insights sells this as a premium feature at $12/acre/year.
**Implementation**: `analyze_yield_gap(crop_id)` computes attainable yield from variety `yield_potential_kg_ha`, scales by area, diffs against actual; returns gap_kg, gap_pct, and ranked `limiting_factors` list derived from input records (fertiliser, water, seed density).
**Competitive reference**: Granular Insights Yield Benchmarking, CIMMYT Yield Gap Atlas

---

### I3. Crop Water Stress Index (CWSI) Monitoring
**Category**: Feature
**Justification**: Water stress is the single largest yield-limiting factor in sub-Saharan Africa and South Asia. Early detection via soil-moisture proxy saves 20–40% water while maintaining yield. Trimble Ag and Lindsay Zimmatic integrate CWSI into irrigation scheduling.
**Implementation**: `record_water_stress_event(crop_id, cwsi_value, measurement_method, sensor_id)` persists events; `get_water_stress_timeline(crop_id)` returns time-series with alert flags when CWSI > 0.5 (moderate) or > 0.8 (severe).
**Competitive reference**: Lindsay Zimmatic FieldNET Advisor, Trimble Ag WaterLink

---

### I4. Input Cost Ledger with ROI Calculation
**Category**: Feature
**Justification**: Crop profitability requires knowing total input cost (seed, fertiliser, pesticide, labour, machinery) per hectare. Most smallholder platforms track yield but not cost, making ROI invisible. Deere's MyJohnDeere and Agworld both monetise this as a farm finance module.
**Implementation**: `record_input_application(crop_id, input_type, quantity, unit, unit_cost_decimal, applied_at)` accumulates per-crop; `calculate_crop_roi(crop_id, output_price_per_kg)` returns `{total_input_cost, gross_revenue, net_margin, roi_pct}` using `Decimal` throughout.
**Competitive reference**: Agworld Input Cost Tracking, MyJohnDeere Financial Summary

---

### I5. Seed Lot Traceability Chain
**Category**: Compliance
**Justification**: Global food safety regulations (EU Farm-to-Fork, USDA NOP, Kenya KEPHIS) require seed-to-shelf traceability. Buyers in export markets require phytosanitary certificates that can only be generated if the seed lot chain is auditable. SAP AgriSolutions and ContractPodAi sell traceability modules at significant margins.
**Implementation**: `register_seed_lot(lot_id, variety_id, supplier, cert_number, germination_pct, quantity_kg, expiry_date)` with hash of cert doc; `get_seed_lot_trace(crop_id)` walks from crop → seed_lot → supplier → certification, returning complete provenance chain.
**Competitive reference**: SAP AgriSolutions Seed Traceability, FoodLogiQ Connect

---

### I6. Multi-Season Yield Trend Analysis
**Category**: AI/ML
**Justification**: Single-season data is noise. Three-to-five season trend lines reveal soil degradation, variety performance drift, and climate adaptation needs. Corteva Agriscience's Granular platform charges for this analytics layer; it drives seed brand loyalty by proving variety superiority over time.
**Implementation**: `get_multi_season_trend(farm_parcel_id, crop_type, metric)` where metric ∈ {yield_kg_ha, input_cost_ha, roi_pct}; returns list of `{season, value, yoy_delta_pct}` and regression slope indicating improvement/decline trajectory.
**Competitive reference**: Granular Farm Management Platform, Corteva Encirca

---

### I7. Pest and Disease Pressure Alerts
**Category**: Feature
**Justification**: Late blight, fall armyworm, and aflatoxin cause 25–40% crop losses annually in developing markets. Rule-based alert engines that fire when temperature-humidity windows match pathogen sporulation conditions are offered by DTN/Progressive Farmer and Syngenta's Cropwise as premium subscriptions.
**Implementation**: `record_pest_observation(crop_id, pest_type, severity, location_gps, observed_at)` with severity ∈ {low, moderate, high, critical}; `get_pest_pressure_summary(farm_parcel_id, season)` aggregates active alerts and triggers `pest.alert` events when severity escalates.
**Competitive reference**: Syngenta Cropwise Protector, DTN Pest Pressure Maps

---

### I8. Agronomic Advisory Rules Engine
**Category**: AI/ML
**Justification**: Prescriptive agronomy (what to do, when, based on current crop state) is the highest-value layer in precision agriculture. AGCO's Fuse platform and Bayer CropScience's Digital Farming suite generate in-season advisories from phenology + weather + pest data, turning data stores into advisors.
**Implementation**: `generate_agronomic_advice(crop_id)` evaluates current growth stage, days since last observation, GDD progress, and pest alerts to emit ranked list of `{action, urgency, rationale, deadline}` recommendations using a deterministic rules table keyed on crop_type × growth_stage.
**Competitive reference**: Bayer Digital Farming xarvio FIELD MANAGER, AGCO Fuse Advisory

---

### I9. Soil Health Score Integration
**Category**: Integration
**Justification**: Crop yield is bounded by soil health. Carbon content, pH, and compaction determine nutrient availability. Indigo Agriculture and Soil Wealth charge premium for soil health scoring linked to crop performance. Integration with agr_soil closes the agronomic loop and supports carbon credit calculations.
**Implementation**: `attach_soil_health_score(farm_parcel_id, soil_organic_carbon_pct, ph, bulk_density, test_date)` persisted per parcel; `get_soil_adjusted_yield_potential(crop_id)` multiplies variety `yield_potential_kg_ha` by a correction factor derived from pH and OC deviation from optimum.
**Competitive reference**: Indigo Carbon Platform, Soil Wealth ICP

---

### I10. Carbon Sequestration Estimation
**Category**: Compliance
**Justification**: Voluntary carbon markets (VCMs) pay $15–50/tonne CO2e for verified soil carbon credits from rotation planning, cover cropping, and reduced tillage. Corteva's Granular Insights and Indigo Carbon both offer carbon estimation as a revenue stream for farmers.
**Implementation**: `estimate_carbon_sequestration(rotation_plan_id)` applies IPCC Tier 1 emission factors per crop type in the sequence; returns `{estimated_co2e_tonnes_per_ha, creditable_practices, methodology}` for use in VCM applications.
**Competitive reference**: Indigo Carbon, Corteva Carbon Program, Bayer Carbon Initiative

---

### I11. Weather-Adjusted Planting Window Scoring
**Category**: Feature
**Justification**: Static planting calendars fail when seasons shift. A dynamic scoring function that adjusts planting windows based on actual onset-of-rains reduces planting timing risk, which accounts for 15–20% of yield variance. This is a core feature of Climate Corporation's FieldView.
**Implementation**: `score_planting_readiness(crop_type, region, current_date, accumulated_rainfall_mm)` returns a 0–100 readiness score with `go_no_go` boolean and `days_until_optimal` estimate by comparing accumulated rainfall against calendar threshold.
**Competitive reference**: Climate FieldView Planting Insights, DTN Weather Risk Score

---

### I12. Harvest Logistics Scheduling
**Category**: Feature
**Justification**: Harvest timing is a logistics problem as much as an agronomy problem — combine availability, drying capacity, and market windows must align. Agrian and AgriWebb integrate harvest scheduling with equipment and labour resources; poor scheduling causes 5–10% post-harvest losses.
**Implementation**: `schedule_harvest(crop_id, earliest_date, latest_date, equipment_ids, priority)` creates a harvest slot record; `get_harvest_schedule(farm_parcel_id, season)` returns sorted slots with conflict detection when equipment_ids overlap across crops on the same date.
**Competitive reference**: AgriWebb Harvest Planner, Agrian Operations Scheduling

---

### I13. Variety Performance Leaderboard
**Category**: Feature
**Justification**: Seed companies and extension officers need to know which varieties perform best per region and season. A ranked leaderboard built from actual yield records creates a data moat and drives variety registry adoption. Pioneer and Syngenta both publish regional trial data — this democratises that intelligence for smallholders.
**Implementation**: `get_variety_leaderboard(crop_type, region, season)` aggregates yield records by variety, computes mean and stddev of `yield_kg_ha`, ranks by mean, and returns top-N entries with confidence intervals for evidence-based variety selection.
**Competitive reference**: Pioneer Performance Trials, Syngenta Regional Variety Trials

---

### I14. Compliance Export: Season KPI Reporting
**Category**: Compliance
**Justification**: NGOs, governments, and agricultural insurers require standardised reporting (FAO GAEZ format, donor KPI templates). Automated exports reduce audit friction and unlock development finance (AGRA, World Bank projects require traceability reports). Manual exports are error-prone and un-auditable.
**Implementation**: `export_season_compliance_report(season, format)` where format ∈ {json, csv}; marshals crops + yields + inputs + rotation data into a structured report with SHA-256 checksum; emits `report.generated` audit event for tamper evidence.
**Competitive reference**: CGIAR Digital Inclusion tools, AGRA M&E Platform

---

### I15. Inter-Capability Event Bus Integration
**Category**: Integration
**Justification**: Crop data is consumed by at least five other APG capabilities: agr_irr (irrigation), agr_frt (fertilisation), agr_mkt (market prices), fin_ins (crop insurance), and agr_soil (soil health). Publishing typed CloudEvents on a shared bus eliminates polling and enables reactive workflows — the same pattern used by SAP Event Mesh.
**Implementation**: `publish_crop_event(event_type, entity_id, payload)` serialises to CloudEvents 1.0 spec with `source=agr_crp/{tenant_id}`, `datacontenttype=application/json`, and writes to an async event queue; other capabilities subscribe by event type prefix `agr_crp.*`.
**Competitive reference**: SAP Event Mesh AgriSolutions, Salesforce Agentforce Platform Events
