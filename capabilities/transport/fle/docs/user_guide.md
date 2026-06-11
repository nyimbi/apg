# Fleet Management — User Guide

**Capability**: `transport_fle` v2.0.0  
**Platform**: APG  
**Audience**: Fleet managers, dispatchers, compliance officers

---

## Overview

The APG Fleet Management capability provides end-to-end lifecycle management for commercial vehicle fleets. It covers vehicle registration, driver management, trip planning and execution, fuel tracking, maintenance scheduling, compliance monitoring, tachograph/HOS enforcement, COF inspections, real-time telematics, incident management, and TCO analytics.

---

## Getting Started

### Dashboard

Navigate to `/fle/` to reach the Fleet Dashboard. Key metrics displayed:

| KPI | Description |
|-----|-------------|
| Total Vehicles | All registered vehicles (non-deleted) |
| Active Vehicles | Vehicles with status `active` |
| On Trip | Vehicles currently `dispatched` or `in_progress` |
| In Maintenance | Vehicles undergoing service |
| Active Drivers | Drivers with status `active` |
| Compliance Alerts | Documents expiring within 30 days |
| Overdue Maintenance | Maintenance items past scheduled date |
| Active Incidents | Incidents not yet resolved/closed |

The dashboard also shows:
- Top 5 predictive maintenance alerts (urgency: critical/high)
- Top 10 critical compliance events (COF, insurance, licence expiry)

---

## Vehicles

### Register a Vehicle

1. Go to **Vehicles → Register Vehicle** (`/fle/vehicles/new`)
2. Fill in:
   - Registration plate (unique per tenant)
   - VIN (minimum 11 chars, unique per tenant)
   - Vehicle type, make, model, year
   - Fuel type, ownership type
   - GVW and payload capacity (used for overloading checks)
3. Click **Register**

### Vehicle Status Lifecycle

```
active → in_maintenance → active
active → out_of_service  (after failed inspection or critical incident)
active → breakdown       (mid-trip breakdown)
active → disposed
active → on_hire
active → awaiting_inspection
```

### Vehicle Detail

The vehicle detail page (`/fle/vehicles/<id>`) shows:
- Current status and odometer
- TCO breakdown (fuel, maintenance, insurance, fines, cost/km)
- Maintenance history
- Recent fuel records
- Last 10 trips
- Last GPS position
- All inspections and incidents

---

## Drivers

### Register a Driver

1. Go to **Drivers → Register Driver** (`/fle/drivers/new`)
2. Provide:
   - Full name, employee number
   - Licence number and class (A–DE EU system)
   - Licence expiry date
   - CPC expiry (Certificate of Professional Competence)
   - Medical certificate expiry (optional)
   - Tachograph card number
3. The system validates:
   - Licence must not be expired on registration
   - CPC must not be expired if provided

### Driver Behaviour Score

Every driver has an automatically computed behaviour score (0–100, grades A–F) based on telematics events:

| Dimension | Weight | Events Counted |
|-----------|--------|----------------|
| Speeding | 25% | `speeding` events |
| Harsh Braking | 20% | `harsh_braking` |
| Harsh Acceleration | 15% | `harsh_acceleration` |
| Cornering | 10% | `harsh_cornering` |
| Seatbelt | 15% | `seatbelt_violation` |
| Distraction | 10% | `distraction` |
| Idle | 5% | `idle` |

Score is calculated per 100 km driven. View via **Drivers → Driver Detail → Behaviour Score**.

---

## Trips

### Plan a Trip

1. Go to **Trips → Plan Trip** (`/fle/trips/new`)
2. Select vehicle and driver
3. Set origin, destination, planned departure/arrival
4. Enter load weight (kg)
5. If cross-border: tick **Customs Required** and list countries

**Pre-departure checks performed automatically:**
- Vehicle must be `active`
- Driver must be `active` with valid licence
- Load must not exceed vehicle payload capacity
- Vehicle must not already be on an active trip
- Driver must not already be on another trip

### Trip Status Transitions

```
PLANNED → DISPATCHED → IN_PROGRESS → COMPLETED
PLANNED → CANCELLED
DISPATCHED → CANCELLED
IN_PROGRESS → BREAKDOWN
IN_PROGRESS → DELAYED
```

### Mid-Trip Operations

**Change driver mid-trip:**  
`POST /api/fle/v1/trips/<id>/change-driver` with `{"new_driver_id": "...", "reason": "..."}`  
Validates new driver's licence and active status.

**Record breakdown:**  
`POST /api/fle/v1/trips/<id>/breakdown`  
Automatically sets vehicle status to `breakdown`.

---

## Fuel Records

Record every fuel fill-up at **Fuel → Record Purchase** or via API:

```json
POST /api/fle/v1/fuel
{
  "vehicle_id": "...",
  "litres": 120.5,
  "cost_per_litre": 185.00,
  "odometer_km": 50000,
  "station_name": "Total Thika Road"
}
```

The system:
- Calculates `total_cost = litres × cost_per_litre`
- Validates odometer is not regressing
- Updates vehicle's current odometer

---

## Maintenance

### Schedule Maintenance

Types: `scheduled`, `corrective`, `predictive`, `emergency`

Corrective maintenance is **automatically scheduled** when an inspection fails — one job per defect found.

### Maintenance Lifecycle

1. **Schedule** → status: `scheduled`
2. **Start** (`POST /maintenance/<id>/start`) → status: `in_progress`, vehicle → `in_maintenance`
3. **Complete** (`POST /maintenance/<id>/complete` + actual_cost) → status: `completed`, vehicle → `active`

### Predictive Maintenance Alerts

View at `/fle/reports/predictive-maintenance`. The system generates alerts based on:
- Oil change intervals (10,000 km or 180 days)
- Overdue scheduled maintenance items
- (Production: feeds telematics sensor data to ML model via APG ai_orchestration)

---

## Inspections & COF

### Pre/Post Trip Inspections

Record at `POST /api/fle/v1/inspections`. If result is `fail`:
- Vehicle automatically set to `out_of_service`
- Corrective maintenance scheduled for each defect
- Fleet manager notified

### Certificate of Fitness (COF)

Kenya/East Africa statutory requirement. Record at `POST /api/fle/v1/cof`.  
COF expiry tracked in compliance calendar — alerts issued at 30 days and 7 days before expiry.

---

## Compliance Calendar

View at `/fle/compliance`. Shows all upcoming compliance events colour-coded by urgency:

| Severity | Condition |
|----------|-----------|
| Critical (red) | Overdue OR ≤7 days |
| Warning (amber) | 8–30 days |
| Info (green) | >30 days |

**Event types tracked:**
- Insurance renewal
- COF renewal
- Vehicle registration renewal
- Driver licence expiry
- Driver CPC expiry
- Scheduled maintenance due

---

## Tachograph / Hours of Service

### EU (EC 561/2006)

The system enforces:
- Max 4h30m continuous driving before mandatory break
- Max 9h daily driving (extendable to 10h twice/week)
- Max 56h weekly driving
- Max 90h fortnightly driving
- Min 11h daily rest

Any submission violating these rules is rejected with a specific infringement code.

### US HOS (49 CFR 395)

- Max 11h driving after 10h off-duty
- 14-hour on-duty window
- 60h/7-day or 70h/8-day cycle

---

## Incidents

Report at `POST /api/fle/v1/incidents`. Business rules:
- Must be reported within 24 hours of occurrence
- `fatal` or `critical` incidents require a police reference number
- `major`, `critical`, `fatal` severity → vehicle automatically set `out_of_service`

### Overloading Fine Allocation

Record `overloading_fine_allocated` on the incident. The domain calculation `allocate_overloading_fine()` splits between driver and owner by configurable percentage.

---

## Insurance & Registration

Register insurance policies and vehicle registration documents. Expiry dates feed the compliance calendar automatically.

---

## Reports

| Report | URL |
|--------|-----|
| Fleet Dashboard KPIs | `GET /api/fle/v1/dashboard` |
| TCO for vehicle | `GET /api/fle/v1/reports/tco/<vehicle_id>` |
| Fleet utilisation | `GET /api/fle/v1/reports/utilisation` |
| Compliance calendar | `GET /api/fle/v1/reports/compliance-calendar` |
| Predictive maintenance | `GET /api/fle/v1/reports/predictive-maintenance` |
| Driver behaviour score | `GET /api/fle/v1/reports/driver-score/<driver_id>` |

---

## Cross-Border Operations

For trips crossing borders:
1. Set `customs_required: true`
2. List `cross_border_countries: ["TZ", "UG"]`
3. The system requires customs documentation to be confirmed before dispatch

---

## Hired/Rental Vehicles

Vehicles with `ownership_type` of `hired` or `contract_hire` are validated against hire period dates on dispatch. Operations outside the hire window are blocked.

---

## Fuel Fraud Audit

Every fuel record can be audited for anomalies via `POST /api/fle/v1/fuel/<id>/audit`.

The engine checks three independent signals:

| Signal | Trigger | Severity |
|--------|---------|----------|
| Price deviation | Cost/litre > 15% above fleet average | Medium |
| Duplicate receipt | Same receipt_ref on another record | High |
| Volume vs expected | Claimed litres > 140% of telematics-derived consumption | High |

Each detected anomaly contributes 0.35 to a `risk_score` (0.0–1.0). Records with `risk_score >= 0.35` emit a `fuel.audit_flagged` event for investigation.

---

## Driver Fatigue Risk Scoring

`GET /api/fle/v1/drivers/<id>/fatigue-risk?lookback_days=7`

Analyses tachograph records over the lookback window and returns:

| Factor | Threshold | Weight |
|--------|-----------|--------|
| Consecutive max-driving days | >= 4 days at 9h+ | 0.30 |
| Repeated split rests | >= 3 times < 11h solid | 0.25 |
| Weekly hours near ceiling | >= 83% of 56h limit | 0.20 |
| Recorded infringements | Any code present | 0.25 |

Risk levels: **low** (< 0.4), **high** (0.4–0.69), **critical** (>= 0.7).  
At critical, the system recommends "Mandate 48h off-duty immediately."

No additional hardware required — uses existing tachograph data.

---

## Vehicle Disposal Recommendations

`GET /api/fle/v1/vehicles/<id>/disposal?market_value=<KES>`

Compares the vehicle's TCO per km against the fleet average and evaluates the maintenance cost trend over the last 6 completed jobs.

| Recommendation | Condition |
|----------------|-----------|
| **replace** | TCO premium > 20% AND payback < 18 months, OR maintenance cost rising > 25% |
| **monitor** | TCO premium 10–20% OR maintenance trend 15–25% |
| **retain** | Within acceptable TCO range |

Payback is calculated as: `(replacement_cost − market_value) ÷ annual_tco_premium`.

---

## Shift Assignment Optimisation

`POST /api/fle/v1/shifts/optimise` with `{"date": "2026-06-15"}`

Assigns available active drivers to planned trips for the given date while respecting EU EC 561/2006 daily driving limits (9h).

The greedy solver:
1. Sorts trips by planned departure time
2. Assigns the least-loaded eligible driver to each trip
3. Returns `feasibility_score` per assignment and flags trips with `no_driver_within_hos_limit`

Production note: replace the greedy solver with Google OR-Tools CP-SAT for global optimality on large fleets.

---

## Budget Burn-Rate Variance

`POST /api/fle/v1/reports/budget-variance` with `{"fuel_budget_month": 500000, "maintenance_budget_month": 200000}`

Returns month-to-date actuals and projects month-end spend from the current burn rate.

| Alert Level | Condition |
|-------------|-----------|
| ok | Projected overspend <= 10% |
| warning | Projected overspend 10–20% |
| critical | Projected overspend > 20% |

A `budget.overspend_alert` event is emitted at warning or critical — consumed by `ntfy` to notify the finance team.

---

## Insurance Claim Pack

`GET /api/fle/v1/incidents/<id>/claim-pack`

Generates a structured evidence package containing:
- Full incident record
- ±30-minute telematics replay (GPS track, speed, events)
- Driver behaviour score (YTD)
- Vehicle TCO summary
- Current COF certificate
- Active insurance policy details
- Last 5 completed maintenance records

The `claim_reference` (e.g. `APG-3F9A1B2C`) can be quoted directly on the insurer's claims portal. Reduces claim preparation time from 4–6 hours to under 2 minutes.

---

## Geofence Workflow Triggers

`POST /api/fle/v1/geofence/event`

```json
{
  "vehicle_id": "...",
  "geofence_id": "customer_site",
  "event_type": "entry",
  "trip_id": "...",
  "geofence_label": "Bamburi Cement Mombasa"
}
```

Pre-configured geofence/event pairs execute workflow steps automatically:

| Geofence Key | Steps Executed |
|--------------|----------------|
| `customer_site_entry` | Notify warehouse + driver, trigger POD workflow |
| `depot_entry` | Notify yard manager, trigger post-trip inspection, check maintenance interval |
| `restricted_zone_entry` | Notify compliance officer, trigger violation workflow |

Steps are emitted as domain events for `wflo` and `ntfy` to process.

---

## Driver Coaching Events

`POST /api/fle/v1/telematics/<telematics_event_id>/coaching`

Generates a contextual in-cab coaching message for a telematics event. Supported event types:

| Event | Message Tone | Priority |
|-------|-------------|----------|
| speeding | Informational | High |
| harsh_braking | Coaching | Medium |
| harsh_acceleration | Coaching | Medium |
| idle | Reminder | Low |
| seatbelt_violation | Mandatory | Critical |
| distraction | Mandatory | Critical |

Returns `null` for events without a registered coaching script.

---

## Vehicle Health Snapshot

`GET /api/fle/v1/vehicles/<id>/health`

Single-call aggregated health view for the vehicle detail page:

| Field | Source |
|-------|--------|
| `health_score` (0–100) | Deducted by overdue compliance, maintenance, incidents, and predictive alerts |
| `last_position` | Latest telematics event |
| `tco_summary` | Total cost, cost/km, distance |
| `critical_compliance_events` | Compliance calendar filtered to this vehicle |
| `predictive_alerts` | ML/rule-based maintenance predictions |
| `open_incidents` | Count of unresolved incidents |

---

## Driver Leaderboard

`GET /api/fle/v1/reports/driver-leaderboard?top_n=10`

Returns the top N active drivers ranked by overall behaviour score descending. Each entry includes rank, score, grade, trip count, distance, and incident count. Use for:
- Monthly performance reviews
- Driver incentive programmes
- Insurance premium negotiations

---

## Deferring Maintenance

`POST /api/fle/v1/maintenance/<id>/defer`

```json
{"new_date": "2026-07-01T08:00:00", "reason": "Parts on order — ETA 25 June"}
```

Moves a `scheduled` or `overdue` maintenance job to a new date with full audit trail (actor, reason, original date preserved in notes). The `maintenance.deferred` event is emitted for downstream notification.
