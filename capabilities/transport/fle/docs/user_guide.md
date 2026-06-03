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
