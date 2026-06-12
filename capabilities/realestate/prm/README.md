# Property Marketing & Management

## Overview
Central portfolio management and property marketing platform for all real estate assets. Registers properties and units, manages owner entities and their distributions, tracks performance KPIs (occupancy, WAULT, yield), coordinates handovers, and drives the full marketing funnel — from listing publication and virtual tour orchestration through to lead capture, scoring, and conversion tracking.

## Capability ID
`realestate_prm`

## Provides
- `property_portfolio_management`: Register and lifecycle-manage properties across types and tiers
- `unit_management`: Create, assign, and track individual lettable units
- `owner_portal_service`: Secure portal with statements, documents, and distribution history
- `property_performance_reporting`: KPI engine for occupancy, void rate, yield, CAPEX ratio
- `portfolio_analytics`: Portfolio-wide benchmarking and trend analysis
- `handover_management`: Structured landlord/developer/contractor handover workflow
- `owner_distribution_management`: Dual-control net distribution payments
- `property_data_room`: Secure document access with mandatory access logging
- `performance_kpi_engine`: On-demand KPI calculation against live unit data
- `property_benchmarking`: Cross-portfolio comparisons
- `listing_management`: Compose, publish, and multi-channel syndicate property listings
- `lead_management`: Capture, score, route, and pipeline-track prospect leads
- `virtual_tour_management`: Register and analytics-track 360°/video virtual tours
- `marketing_funnel_analytics`: Funnel reporting from listing views through to lease conversion

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | User identity and permissions |
| `audl` | Audit trail for status changes and distributions |
| `mten` | Tenant isolation for portfolio data |
| `conf` | Portfolio-level configuration |
| `ntfy` | Void unit alerts, occupancy threshold alerts, lead SLA alerts |
| `wflo` | Board approval workflow for deletions |
| `moni` | Occupancy threshold monitoring |
| `mqeb` | Publish unit status changes to stream |
| `srch` | Full-text property search |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| `portfolio.supported_tiers` | core, core_plus, … | Valid portfolio classifications |
| `governance.data_room_access_always_logged` | true | Mandatory access logging |
| `governance.dual_control_for_distributions` | true | Two approvers for owner payments |
| `marketing.lead_urgency_decay_days` | 14 | Days before urgency score decays without contact |
| `marketing.listing_channels` | website | Default publication channels |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/prm/dashboard` | GET | Portfolio summary | `view` |
| `/realestate/prm/properties` | GET/POST | List/register properties | `properties` |
| `/realestate/prm/properties/<id>` | GET/PUT/DELETE | CRUD | `properties` |
| `/realestate/prm/properties/search` | GET | Text search | `properties` |
| `/realestate/prm/units` | GET/POST | List/create units | `units` |
| `/realestate/prm/units/void` | GET | Void units | `units` |
| `/realestate/prm/owners` | GET/POST | List/register owners | `owners` |
| `/realestate/prm/kpis` | POST | Calculate KPIs | `kpis` |
| `/realestate/prm/distributions` | GET/POST | List/create distributions | `distributions` |
| `/realestate/prm/distributions/<id>/approve` | POST | Dual-control approval | `distributions` |
| `/realestate/prm/handovers` | POST | Create handover | `handovers` |
| `/realestate/prm/listings` | GET/POST | List/publish listings | `listings` |
| `/realestate/prm/listings/<id>/unpublish` | POST | Deactivate listing | `listings` |
| `/realestate/prm/leads` | GET/POST | List/capture leads | `leads` |
| `/realestate/prm/leads/<id>/assign` | POST | Assign lead to agent | `leads` |
| `/realestate/prm/leads/<id>/status` | PUT | Advance lead status | `leads` |
| `/realestate/prm/tours` | POST | Create virtual tour | `tours` |
| `/realestate/prm/tours/<id>/view` | POST | Record tour view event | `tours` |
| `/realestate/prm/marketing/funnel` | GET | Marketing funnel report | `analytics` |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `property_deletion_requires_board_approval` | not board approved | deny |
| `sold_property_modification_denied` | status = sold | deny |
| `distribution_requires_dual_control` | same approver | deny |
| `data_room_access_always_logged` | access not logged | deny |
| `kpi_requires_verified_data` | data not verified | deny |
| `cross_tenant_property_denied` | cross-tenant | deny |
| `listing_requires_price_or_rent` | no asking_price and no asking_rent | deny |
| `lead_source_must_be_known` | source not in allowed set | deny |
| `tour_media_type_must_be_supported` | unsupported media_type | deny |

## Data Models
- `OwnerCreate/Response` — owner entity with type, bank details, linked properties
- `PropertyCreate/Response` — property with address, type, ownership structure, portfolio tier
- `UnitCreate/Response` — lettable unit with area, type, and current tenancy link
- `KpiCalculationRequest/Response` — on-demand KPI calculation with per-metric results
- `DistributionCreate/Response` — net owner distribution with dual-control approval
- `HandoverCreate/Response` — structured handover with checklist items
- `Listing` (dict) — composed listing with headline, media, channels, and status
- `Lead` (dict) — prospect with source attribution, urgency score, and pipeline status
- `VirtualTour` (dict) — tour asset with scene URLs, dwell-time stats, and shareable URL

## Streaming Events
- `property_registered`, `property_status_changed`, `property_sold`
- `unit_status_changed`, `unit_let`, `unit_vacated`
- `owner_registered`, `owner_distribution_paid`
- `performance_kpi_calculated`, `occupancy_threshold_breached`
- `handover_completed`, `portfolio_benchmark_generated`
- `listing_published`, `listing_unpublished`, `listing_superseded`
- `lead_captured`, `lead_assigned`, `lead_converted`, `lead_lost`
- `virtual_tour_created`, `virtual_tour_viewed`

## Edge Cases Handled
- Sold property is rendered immutable — all writes rejected
- Board approval enforced at service layer, not just rule engine
- Distributions require two distinct approvers (same user rejected)
- Data room access logging is a hard gate, not advisory
- KPI calculation handles zero-unit portfolios without division error
- Property deletion cascades from owner's `property_ids` list
- Publishing a new listing auto-supersedes the previous active listing for the same (property_id, unit_id)
- Virtual tour auto-links to the active listing on creation
- Funnel rate calculations guard against division by zero at every stage

## Composability Notes
- Is the anchor for `realestate_lea`, `realestate_ren`, `realestate_mai` (all reference property_id)
- Feeds occupancy data to `realestate_val` for income-based valuations
- Owner distributions consume net rental income from `realestate_acc`
- Lead conversion events feed `realestate_lea` to initiate lease drafting
- Virtual tour dwell-time signals feed lead urgency scores for smarter agent prioritisation

## World-Class Enhancements (v2.0)

1. **AI-Powered Property Valuation Engine** — Ollama LLM (`llama3.2`) produces confidence-scored AVM outputs with supporting rationale from live comps data.
2. **Virtual Tour Orchestration Pipeline** — Ingests 360°/video assets, AI-labels scenes via `llava`, generates interactive tour manifests, and feeds dwell-time to lead scoring.
3. **Intelligent Lead Capture and Scoring** — ML urgency scores, deduplication, source attribution, agent auto-routing with SLA timers and score decay.
4. **Dynamic Listing Publication Engine** — Multi-channel (website, portals, WhatsApp) listing composer with freshness TTLs and automatic unpublish on status change.
5. **Geospatial Search and Radius Queries** — PostGIS-backed `search_properties_near(lat, lng, radius_km)`, isochrone filters, heatmap export.
6. **Tenant Demand Forecasting** — Prophet/ARIMA time-series model forecasts occupancy for next 3/6/12 months with confidence bands.
7. **Lease Expiry and Break-Clause Management** — `lease_expiry_pipeline` categorises active leases by urgency horizon and auto-fires `ntfy` alerts.
8. **Service Charge Reconciliation with Actuals** — Variance analysis per budget line, over/under-recovery apportionment, and certified Section 20B statements.
9. **Maintenance-Integrated CAPEX Tracking** — Links `realestate_mai` work orders to CAPEX ledger with depreciation schedules and capitalisation accounting.
10. **Owner Portal with Document Data Room** — `PortalService` aggregates statements, KPIs, and certificates in a hard-logged document vault.
11. **Bulk Listing Import and Validation Pipeline** — CSV/XLSX batch import with schema validation, local geocoding, fuzzy deduplication, and per-row error reports.
12. **Market Comparables and Benchmarking API** — `get_market_comparables` returns ERV, passing rent, void incentives, and transaction yields for peer properties.
13. **Automated Compliance Checklist Engine** — Per-property, per-jurisdiction checklist (EPC, fire, gas, legionella) with expiry tracking and letting block on critical failures.
14. **Streaming Event Enrichment via CloudEvents** — All status changes published as typed `CloudEvent` payloads to `mqeb` with full correlation IDs for downstream fan-out.
15. **Multi-Currency and FX Rate Integration** — `CurrencyService` fetches live FX rates, converts owner statements to reporting currency (KES/USD/GBP) per ISO 4217.

## New Methods

### `capture_lead` — Intake a prospect with source attribution and urgency scoring

```python
lead = await svc.capture_lead(
    tenant_id="t1",
    property_id="prop_abc",
    unit_id="unit_01",
    contact={"name": "Alice Kamau", "email": "alice@example.com", "phone": "+254700000001"},
    source="virtual_tour",          # portal | referral | direct | virtual_tour
    notes="Interested in 2BR unit, move-in next month",
)
# Returns dict with lead_id, urgency_score, assigned_agent, sla_due_at
```

### `publish_listing` — Compose and syndicate a listing to multiple channels

```python
listing = await svc.publish_listing(
    tenant_id="t1",
    property_id="prop_abc",
    unit_id="unit_01",
    headline="Spacious 2BR in Westlands — Available Now",
    asking_rent=Decimal("85000"),
    currency="KES",
    channels=["website", "portal", "whatsapp"],
    media_urls=["https://cdn.example.com/img1.jpg"],
    freshness_days=30,              # auto-unpublishes after 30 days
)
# Returns dict with listing_id, published_at, channel_results, shareable_url
```

### `create_virtual_tour` — Register a 360°/video tour and link it to the active listing

```python
tour = await svc.create_virtual_tour(
    tenant_id="t1",
    property_id="prop_abc",
    unit_id="unit_01",
    media_type="360_images",        # 360_images | video_walkthrough
    scene_urls=["https://cdn.example.com/tour/room1.jpg", "https://cdn.example.com/tour/room2.jpg"],
    auto_label_scenes=True,         # invokes local llava vision model
)
# Returns dict with tour_id, shareable_url, listing_id (auto-linked), scene_labels
```
