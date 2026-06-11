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
