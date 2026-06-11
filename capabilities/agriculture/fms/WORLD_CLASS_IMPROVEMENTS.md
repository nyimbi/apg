# Farm Management System — World-Class Improvements

15 targeted improvements to make agr_fms a competitive agricultural data platform.

---

### I1. Yield Recording and Cost-per-Kg Analysis
**Category**: Feature
**Justification**: Without yield data, cost-per-ha is a vanity metric. Linking harvest tonnage to input/labour spend gives the grower the single most actionable number in agriculture — gross margin per hectare — which John Deere Operations Center surfaces as the primary KPI. Competitors without this force manual spreadsheet reconciliation.
**Implementation**: Add `_harvests` store; `create_harvest_record(parcel_id, crop_id, quantity_kg, sale_price_per_kg)` derives `revenue`, `gross_margin`, and `cost_per_kg` using Decimal arithmetic; `get_gross_margin_report(parcel_id)` joins harvest against parcel cost summary.
**Competitive reference**: John Deere Operations Center — "Field Profitability" dashboard

---

### I2. Crop Season / Campaign Tracking
**Category**: Feature
**Justification**: All inputs, labour, and diary entries belong to a season (Long Rains 2025, Season A 2026). Without seasons, cost roll-ups conflate multiple growing cycles and make year-on-year comparison impossible. Trimble Ag Software separates every cost by campaign.
**Implementation**: Add `_seasons` store keyed by `(parcel_id, season_name)`; stamp each input and labour record with `season_id`; add `list_season_costs(season_id)` and `compare_seasons(season_id_a, season_id_b)` that return a diff dict.
**Competitive reference**: Trimble Ag Software — season-aware cost templates

---

### I3. Agrochemical Compliance Ledger (Pre-Harvest Interval Enforcement)
**Category**: Compliance
**Justification**: Applying a pesticide within its pre-harvest interval (PHI) is an export-market disqualifier and a food-safety liability. No basic FMS enforces PHI. Kenya Plant Health Inspectorate Board (KEPHIS) and GlobalG.A.P. audits demand a compliant ledger. This alone can unlock premium buyer contracts.
**Implementation**: Store `phi_days` per chemical input; `check_phi_compliance(parcel_id, harvest_date)` returns a list of inputs whose `applied_date + phi_days > harvest_date`, flagging non-compliant chemicals with the violation window in days.
**Competitive reference**: Croptracker — PHI compliance reports for GAP certification

---

### I4. Soil Nutrient Balance Tracking
**Category**: AI/ML
**Justification**: Soil nutrient depletion is silent until yield collapses. Tracking N-P-K applied vs crop offtake lets the system recommend topdress schedules before deficiency symptoms appear — a capability that FARMDOK charges a premium tier for.
**Implementation**: Maintain a per-parcel `_nutrient_ledger` summing N/P/K kg/ha from fertiliser inputs (using product lookup table) minus crop offtake constants; `get_nutrient_balance(parcel_id)` returns balance and a traffic-light status (`surplus|adequate|deficit`).
**Competitive reference**: FARMDOK — nutrient balance and fertilisation planning module

---

### I5. Weather Event Correlation in Diary
**Category**: Integration
**Justification**: Agronomic decisions (spray windows, irrigation triggers) are meaningless without weather context. Linking diary entries to weather snapshots (rainfall mm, temperature, humidity) enables retrospective analysis of input efficacy vs conditions — what Climate Corporation built its $1B valuation on.
**Implementation**: `create_diary_entry` accepts optional `weather_snapshot: dict` (rain_mm, temp_c, humidity_pct, wind_kph); `get_diary_weather_summary(parcel_id, from_date, to_date)` aggregates total rainfall and average temperatures across diary entries for the period.
**Competitive reference**: The Climate Corporation (Climate FieldView) — weather-linked field scouting

---

### I6. Labour Contractor / Worker Registry
**Category**: Feature
**Justification**: Most smallholder farms use casual labour hired through named gang leaders (contractors). Recording individual workers enables traceability for safety incidents, overtime tracking, and payroll integration — a gap Agworld explicitly fills for compliance-sensitive operations.
**Implementation**: Add `_workers` store; `register_worker(name, id_number, contractor_id, skill_tags)` returns a worker record; labour schedule accepts `worker_ids: list[str]`; `get_worker_payroll_summary(worker_id, from_date, to_date)` computes total days worked and gross pay.
**Competitive reference**: Agworld — workforce management and contractor tracking

---

### I7. Irrigation Water Usage Tracking
**Category**: Feature
**Justification**: Water is the most cost-variable input on irrigated farms. Tracking volume applied per parcel per period supports water permit compliance, cost allocation, and efficiency benchmarking. Netafim's FarmView tracks irrigation events as first-class records.
**Implementation**: Add `_irrigation_events` store; `record_irrigation(parcel_id, volume_m3, method, duration_h, cost_per_m3)` stores the event with Decimal cost; `get_irrigation_summary(parcel_id)` returns total m3, cost, and mm-equivalent per hectare.
**Competitive reference**: Netafim FarmView — irrigation scheduling and water cost tracking

---

### I8. Parcel GeoJSON Boundary Storage and Area Validation
**Category**: Feature
**Justification**: Lat/lng centroid is insufficient for precision agriculture. Storing parcel polygon boundaries enables map visualisation, overlap detection (preventing double-counting), and area auto-calculation cross-checked against claimed ha — a basic capability in every tier-1 FMS.
**Implementation**: `update_parcel_boundary(parcel_id, geojson_polygon: dict)` stores the GeoJSON feature; `validate_parcel_area(parcel_id)` uses the shoelace formula (implemented in pure Python) to compute polygon area in ha and returns `{"declared_ha": x, "computed_ha": y, "delta_pct": z}`.
**Competitive reference**: Ag Leader Technology — geospatial parcel management

---

### I9. Input Batch / Lot Traceability
**Category**: Compliance
**Justification**: Food safety recalls require knowing exactly which bag of seed or agrochemical was applied to which parcel on which date. Lot number tracking is a GlobalG.A.P. Annex requirement and a non-negotiable for export supply chains. Croptracker built its entire go-to-market around this.
**Implementation**: `create_input` accepts optional `batch_number`, `expiry_date`, `manufacturer_code`; `trace_input_batch(batch_number)` returns all parcels and dates where that batch was applied, enabling rapid recall scope assessment.
**Competitive reference**: Croptracker — lot traceability and food safety recall management

---

### I10. Budget vs Actual Variance Reporting
**Category**: Feature
**Justification**: Real farm management requires planning a seasonal budget and measuring deviation. Without budget-vs-actual, farmers discover overspend only at season end. Granular variance (by category, by parcel) is the core of AgriWebb's financial planning module.
**Implementation**: `set_season_budget(parcel_id, season_id, budget_by_category: dict[str, Decimal])` persists a budget plan; `get_budget_variance(parcel_id, season_id)` computes `actual - budget` per category and flags categories over threshold with a `status: over|under|on_track` classification.
**Competitive reference**: AgriWebb — budget planning and variance analysis

---

### I11. Automated Reorder Alerts for Input Inventory
**Category**: Feature
**Justification**: Running out of fertiliser mid-season causes yield losses. Tracking on-hand stock vs consumption rate and raising a reorder alert when stock falls below a reorder point is the difference between reactive and predictive farm management — a feature Farmobile prioritises.
**Implementation**: Add `_inventory` store keyed by `(tenant_id, product_name)`; `adjust_inventory(product_name, quantity_delta, unit)` keeps a running balance; `check_reorder_alerts()` returns products where `on_hand_qty <= reorder_point`, sorted by criticality (days_of_supply remaining).
**Competitive reference**: Farmobile — input inventory management with reorder triggers

---

### I12. Crop Rotation History and Rotation Compliance
**Category**: Feature
**Justification**: Continuous monoculture degrades soil and builds pest pressure. Recording which crop was grown in each parcel by season and flagging rotation rule violations (e.g., no legume after legume, minimum 2-year cereal break) is a soil health feature unique to premium FMS platforms.
**Implementation**: `record_crop_season(parcel_id, season_id, crop_name, crop_family)` appends to `_crop_history`; `check_rotation_compliance(parcel_id, rotation_rules: list[dict])` evaluates the last N seasons against rules and returns violations with agronomic explanation strings.
**Competitive reference**: Trimble Ag Software — crop rotation tracking and soil health scoring

---

### I13. Task Due-Date Escalation and Overdue Detection
**Category**: UX
**Justification**: Labour schedules with past due dates that are not completed represent operational failures. Surfacing overdue tasks with days-overdue count and estimated cost impact (delayed harvest, crop loss) turns passive scheduling into active farm operations management — the core of Conservis's daily work plan.
**Implementation**: `get_overdue_tasks(as_of_date: str | None = None)` scans all labour schedules where `scheduled_date < as_of_date and not completed`, returning records enriched with `days_overdue: int` and sorted by urgency; `get_upcoming_tasks(horizon_days: int = 7)` returns tasks due within the horizon.
**Competitive reference**: Conservis — daily work plan with overdue task escalation

---

### I14. Multi-Parcel Bulk Operation Broadcasting
**Category**: UX
**Justification**: Spray programmes, fertiliser top-dresses, and harvest operations typically cover multiple parcels simultaneously. Forcing per-parcel entry for 30-parcel operations is a UX failure that drives users back to spreadsheets. AgriWebb solves this with mob-move-style bulk actions.
**Implementation**: `bulk_create_inputs(parcel_ids: list[str], input_template: dict[str, Any])` iterates parcel_ids, calls `create_input` for each with `farm_parcel_id` substituted, and returns a `{created: [...], errors: [...]}` result dict enabling partial-success handling.
**Competitive reference**: AgriWebb — bulk mob/paddock actions for multi-parcel operations

---

### I15. Export-Ready Compliance Report Generation
**Category**: Compliance
**Justification**: GlobalG.A.P., USAID/AGRA donor reporting, and export buyer audits all require structured evidence packages: input registers, spray records, labour records, and cost summaries in specific formats. Manual PDF assembly costs smallholder cooperatives days per audit cycle. This feature alone closes enterprise contracts.
**Implementation**: `generate_compliance_report(parcel_id, season_id, report_type: str)` assembles a structured dict containing input register (with batch numbers and PHI status), labour register, cost summary, and crop history, tagged with `report_type` (globalg.a.p.|agra|buyer_audit) and a generation timestamp.
**Competitive reference**: Croptracker — one-click GAP compliance report export
