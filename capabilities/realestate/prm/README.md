# Property Management

## Overview
Central portfolio management for all real estate assets. Registers properties and units, manages owner entities and their distributions, tracks performance KPIs (occupancy, WAULT, yield), coordinates handovers, and provides an owner portal and searchable data room for each property.

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

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | User identity and permissions |
| `audl` | Audit trail for status changes and distributions |
| `mten` | Tenant isolation for portfolio data |
| `conf` | Portfolio-level configuration |
| `ntfy` | Void unit alerts, occupancy threshold alerts |
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

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `property_deletion_requires_board_approval` | not board approved | deny |
| `sold_property_modification_denied` | status = sold | deny |
| `distribution_requires_dual_control` | same approver | deny |
| `data_room_access_always_logged` | access not logged | deny |
| `kpi_requires_verified_data` | data not verified | deny |
| `cross_tenant_property_denied` | cross-tenant | deny |

## Data Models
- `OwnerCreate/Response` — owner entity with type, bank details, linked properties
- `PropertyCreate/Response` — property with address, type, ownership structure, portfolio tier
- `UnitCreate/Response` — lettable unit with area, type, and current tenancy link
- `KpiCalculationRequest/Response` — on-demand KPI calculation with per-metric results
- `DistributionCreate/Response` — net owner distribution with dual-control approval
- `HandoverCreate/Response` — structured handover with checklist items

## Streaming Events
- `property_registered`, `property_status_changed`, `property_sold`
- `unit_status_changed`, `unit_let`, `unit_vacated`
- `owner_registered`, `owner_distribution_paid`
- `performance_kpi_calculated`, `occupancy_threshold_breached`
- `handover_completed`, `portfolio_benchmark_generated`

## Edge Cases Handled
- Sold property is rendered immutable — all writes rejected
- Board approval enforced at service layer, not just rule engine
- Distributions require two distinct approvers (same user rejected)
- Data room access logging is a hard gate, not advisory
- KPI calculation handles zero-unit portfolios without division error
- Property deletion cascades from owner's `property_ids` list

## Composability Notes
- Is the anchor for `realestate_lea`, `realestate_ren`, `realestate_mai` (all reference property_id)
- Feeds occupancy data to `realestate_val` for income-based valuations
- Owner distributions consume net rental income from `realestate_acc`
