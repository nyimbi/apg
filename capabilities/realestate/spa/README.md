# Space Planning & Management

## Overview
Comprehensive workplace and space management: versioned floor plans, space allocation and deallocation, move management with headcount-threshold approvals, conflict-checked space bookings, anonymised sensor-data ingestion for occupancy analytics, workplace density planning, and space chargeback calculation.

## Capability ID
`realestate_spa`

## Provides
- `floor_plan_management`: Versioned floor plans in 8 formats (DWG, DXF, IFC, Revit, etc.)
- `space_allocation_engine`: Permanent, hot-desk, shared, and project allocation types
- `move_management_workflow`: Approval-gated moves with churn reason tracking
- `occupancy_analytics`: Multi-sensor occupancy metrics with anonymisation enforcement
- `workplace_density_planning`: Density band targets with 5 workplace strategy types
- `space_booking_engine`: Conflict-checked desk, room, parking, locker bookings
- `sensor_integration_bridge`: Occupancy sensors, badge readers, WiFi probes, AI cameras
- `department_space_reporting`: Space allocation by department with area and headcount
- `space_optimisation_advisor`: Evidence-based recommendations from occupancy history
- `chargeback_space_accounting`: Cost-per-sqm chargeback to departments

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | Move approval authority |
| `audl` | Space allocation audit trail |
| `mten` | Tenant isolation |
| `conf` | Density targets and booking limits |
| `ntfy` | Density threshold alerts |
| `wflo` | Large-move approval workflow |
| `moni` | Real-time occupancy monitoring |
| `mqeb` | Publish space events |
| `schd` | Scheduled occupancy reporting |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| `moves.large_move_headcount_threshold` | 20 | Headcount requiring management approval |
| `bookings.max_advance_booking_days` | 90 | Maximum advance booking window |
| `occupancy.anonymise_sensor_data` | true | Mandate anonymisation before ingestion |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/spa/floor-plans` | GET/POST | Floor plans | `floor_plans` |
| `/realestate/spa/spaces` | GET/POST | Space registry | `spaces` |
| `/realestate/spa/spaces/available` | GET | Available spaces | `spaces` |
| `/realestate/spa/allocations` | GET/POST | Allocations | `allocations` |
| `/realestate/spa/allocations/<id>` | DELETE | Deallocate | `allocations` |
| `/realestate/spa/moves` | GET/POST | Moves | `moves` |
| `/realestate/spa/moves/<id>/approve` | POST | Approve move | `moves` |
| `/realestate/spa/bookings` | GET/POST | Bookings | `bookings` |
| `/realestate/spa/bookings/<id>` | DELETE | Cancel booking | `bookings` |
| `/realestate/spa/occupancy` | POST | Ingest sensor data | `occupancy` |
| `/realestate/spa/occupancy/<property_id>` | GET | Occupancy metrics | `occupancy` |
| `/realestate/spa/density` | POST | Density plan | `density` |
| `/realestate/spa/density/<property_id>` | GET | Density analysis | `density` |
| `/realestate/spa/chargeback/<property_id>` | GET | Chargeback calc | `chargeback` |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `space_double_booking_denied` | time overlap | deny |
| `decommissioned_space_booking_denied` | decommissioned | deny |
| `large_move_requires_approval` | headcount >= 20 | deny |
| `sensor_data_must_be_anonymised` | not anonymised | deny |
| `chargeback_requires_verified_data` | not verified | deny |
| `booking_advance_limit_enforced` | > 90 days ahead | deny |
| `optimisation_requires_occupancy_history` | < 4 weeks data | deny |

## Data Models
- `FloorPlanCreate/Response` — versioned floor plan with format, total area, space list
- `SpaceCreate/Response` — space with type, capacity, area, and status
- `SpaceAllocationCreate/Response` — allocation with type, department, occupants, headcount
- `MoveCreate/Response` — move with from/to spaces, headcount, status
- `BookingCreate/Response` — time-ranged booking with conflict detection
- `OccupancyDataCreate/Response` — anonymised sensor reading with count
- `DensityPlanCreate/Response` — density band target with workplace strategy

## Streaming Events
- `space_registered`, `space_status_changed`, `space_allocated`, `space_deallocated`
- `move_created`, `move_completed`, `move_cancelled`
- `booking_created`, `booking_cancelled`
- `occupancy_data_ingested`, `density_threshold_breached`
- `floor_plan_updated`, `space_chargeback_calculated`

## Edge Cases Handled
- Double-booking detection scans confirmed bookings only (cancelled ignored)
- Floor plan upload auto-increments version for same property+floor
- Small moves (< 20 headcount) auto-approve to `scheduled` status
- Sensor data anonymisation enforced at ingestion, not just configuration
- Booking end < start rejected at Pydantic model level
- Chargeback returns per-department breakdown plus total

## Composability Notes
- Floor plans link to `realestate_prm` property and unit records
- Space chargeback feeds into `realestate_acc` cost allocation runs
- Move completions may trigger `realestate_prm` unit status updates
- Occupancy analytics inform `realestate_val` DCF rental growth assumptions
