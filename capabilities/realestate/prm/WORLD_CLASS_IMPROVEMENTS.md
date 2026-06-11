# Property Marketing (realestate_prm) — World-Class Improvements

**Capability**: `realestate_prm` — Property Listings, Lead Management, Virtual Tours
**Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. AI-Powered Property Valuation Engine

Integrate a locally hosted Ollama LLM (e.g. `llama3.2`) to analyse comparable sales data, rental income, cap rates, and market indices in real time. The valuation engine should produce a confidence-scored AVM (Automated Valuation Model) output with supporting rationale, replacing periodic RICS-only cycles with continuous mark-to-market updates. This feeds directly into `realestate_val` and the owner portal.

## 2. Virtual Tour Orchestration Pipeline

Add a `virtual_tour` subsystem that ingests 360° image sets or video walkthroughs, generates an interactive tour manifest (hotspots, floor plan overlay, room metadata), and publishes a shareable URL. The pipeline should handle upload chunking, AI-assisted scene labelling via a local vision model (e.g. `llava`), and CDN-optimised delivery. Viewers' dwell-time per room becomes a lead-scoring signal.

## 3. Intelligent Lead Capture and Scoring

Build a `LeadManagement` sub-service with intake, deduplication, source attribution (portal / referral / direct / virtual tour), and an ML-scored urgency rating. Leads should auto-route to agents based on property match criteria, with SLA timers and automated follow-up sequencing via the `ntfy` capability. Score decay over time without contact prevents stale leads polluting pipeline metrics.

## 4. Dynamic Listing Publication Engine

Implement a `ListingService` that composes property data, unit availability, media assets, and pricing into a structured listing payload, then publishes it to multiple channels (website, portals, WhatsApp broadcast) via adapters. Listings should carry freshness TTLs — auto-unpublished when a unit status changes to `let` or `not_available`, keeping all channels consistent without manual intervention.

## 5. Geospatial Search and Radius Queries

Replace the current naive text search with PostGIS-backed geospatial queries. Add `lat/lng` indexing to `PropertyAddress`, support radius search (`search_properties_near(lat, lng, radius_km)`), isochrone filters (walk / drive / transit time), and heatmap data export. This unlocks location-intelligence features in the owner portal and prospect-facing search UI.

## 6. Tenant Demand Forecasting

Add a `demand_forecast` async method that uses time-series data (historical void rates, enquiry volumes, seasonality) fed to a lightweight Prophet or ARIMA model served via a local inference endpoint. Output: predicted occupancy for the next 3/6/12 months per property with upper/lower confidence bands. Informs pricing strategy and refurbishment scheduling.

## 7. Lease Expiry and Break-Clause Management

Introduce `lease_expiry_pipeline` that scans active leases (from `realestate_lea`), categorises them by urgency (0–3, 3–6, 6–12 month expiry horizons), and produces a renewal risk score per unit. Automated alerts via `ntfy` fire at configurable thresholds. This converts a reactive diary into a proactive asset management instrument.

## 8. Service Charge Reconciliation with Actuals

Extend `service_charge_budget` to accept actual expenditure entries and produce a variance analysis per budget line. Add a `reconcile_service_charges` method that calculates over/under-recovery, apportions surpluses or deficits across leaseholders by share of occupied area, and generates a certified statement ready for Section 20B (or equivalent statutory) notices.

## 9. Maintenance-Integrated Capex Tracking

Link `realestate_mai` work orders to the property's CAPEX ledger. Add `record_capex_expenditure` and `get_capex_summary` methods. Each expenditure carries an asset category, depreciation schedule, and improvement/maintenance distinction. This enables accurate capitalisation accounting, yield impact modelling, and regulatory compliance for institutional owners.

## 10. Owner Portal with Document Data Room

Add a `PortalService` that aggregates owner statements, KPI dashboards, tax certificates, insurance schedules, and inspection reports into a secure, tenant-scoped document vault. Access events are hard-logged (already in business rules but not implemented). Add `get_data_room_documents`, `upload_document`, and `log_document_access` methods with mandatory audit trail writes.

## 11. Bulk Listing Import and Validation Pipeline

Implement `bulk_import_listings` accepting CSV/XLSX payloads. The pipeline validates each row against field schemas, resolves address geolocation via a local geocoding service, deduplicates against existing records using fuzzy matching, and returns a structured import report (accepted / rejected / duplicate counts with per-row errors). Reduces onboarding time for portfolios migrated from legacy PMS systems.

## 12. Market Comparables and Benchmarking API

Add `get_market_comparables(property_id, radius_km, asset_class)` that queries an internal or external comparable transactions dataset and returns ERV (Estimated Rental Value), passing rent, void incentives, and transaction yields for peer properties. Results feed the AVM engine (#1) and enable agents to justify asking rents with data-backed evidence.

## 13. Automated Compliance Checklist Engine

Replace the current stub `compliance_audit` with a structured engine: per-property, per-jurisdiction compliance checklist (fire safety certificates, EPC ratings, gas safety records, legionella risk assessments). Track certificate expiry dates, auto-schedule renewals, and block unit lettings until critical certificates are current. Compliance state is exposed as a risk score on the owner dashboard.

## 14. Streaming Event Enrichment via CloudEvents

All status-change events currently logged to Python `log` should be published as typed `CloudEvent` payloads to the `mqeb` message bus. Each event should carry `property_id`, `unit_id`, `tenant_id`, `actor_id`, `old_value`, `new_value`, and a correlation ID. Downstream capabilities (`realestate_val`, `realestate_acc`, `realestate_lea`) subscribe to these streams rather than polling, decoupling the entire estate management mesh.

## 15. Multi-Currency and FX Rate Integration

The current service hardcodes `currency = "KES"` as default. Implement a `CurrencyService` shim that fetches live FX rates from a local cache (refreshed by a background job), converts financial figures in KPI outputs and owner statements to the owner's reporting currency, and applies consistent rounding per ISO 4217. This is essential for institutional owners with cross-border portfolios reporting in USD or GBP.
