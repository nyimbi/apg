# Property Management System (hos_pms)

Room inventory, check-in/out, housekeeping, folio management, night audit, group bookings, dynamic pricing, loyalty points, maintenance work orders, rate plan catalogue, and fiscal compliance receipts.

**Version**: 2.0.0 | **Domain**: hospitality

---

## Feature List

### Core (v1)
- Room inventory with status lifecycle (available / occupied / maintenance / housekeeping / out_of_order)
- Guest profiles with VIP tiers, stay counts, and ID document storage
- Reservations: create, modify, cancel with automatic balance recalculation
- Check-in / check-out with room status transitions
- Early check-in and late check-out with optional fee posting
- Folio management: post, void, and summarise charges by type
- Payment posting against open folios
- Housekeeping tasks: create, assign, complete, cancel
- Group booking blocks with tentative/confirmed/cancelled lifecycle
- Night audit: occupancy, arrivals, departures, no-shows, room and ancillary revenue
- Property dashboard summary
- Full audit event log

### World-Class Additions (v2)
| # | Feature | Method(s) |
|---|---------|-----------|
| I1 | Dynamic Pricing Engine | `get_dynamic_rate`, `configure_pricing` |
| I2 | RevPAR / Yield Analytics | `get_revpar_analytics` |
| I5 | Loyalty Points | `accrue_loyalty_points`, `redeem_loyalty_points` |
| I6 | Overbooking Walk Management | `walk_reservation` |
| I7 | Maintenance Work Orders | `create_work_order`, `assign_work_order`, `start_work_order`, `close_work_order`, `list_work_orders` |
| I9 | Rate Plan Catalogue | `create_rate_plan`, `get_applicable_rate`, `list_rate_plans` |
| I12 | Fiscal Compliance Receipts | `post_fiscal_receipt`, `get_fiscal_receipt`, `list_fiscal_receipts` |
| I13 | Auto Housekeeping Assignment | `auto_assign_housekeeping` |

---

## API Endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/pms/health | Service health check |
| GET | /api/hospitality/pms/describe | Capability descriptor |
| GET | /api/hospitality/pms/rooms | List rooms |
| POST | /api/hospitality/pms/rooms | Create room |
| GET | /api/hospitality/pms/rooms/{id} | Get room |
| PUT | /api/hospitality/pms/rooms/{id} | Update room |
| DELETE | /api/hospitality/pms/rooms/{id} | Delete room |
| GET | /api/hospitality/pms/rooms/availability | Availability search |
| GET | /api/hospitality/pms/guests | List guests |
| POST | /api/hospitality/pms/guests | Create guest |
| GET | /api/hospitality/pms/guests/{id} | Get guest |
| PUT | /api/hospitality/pms/guests/{id} | Update guest |
| GET | /api/hospitality/pms/reservations | List reservations |
| POST | /api/hospitality/pms/reservations | Create reservation |
| GET | /api/hospitality/pms/reservations/{id} | Get reservation |
| PUT | /api/hospitality/pms/reservations/{id} | Update reservation |
| DELETE | /api/hospitality/pms/reservations/{id} | Cancel reservation |
| POST | /api/hospitality/pms/reservations/{id}/check-in | Check in |
| POST | /api/hospitality/pms/reservations/{id}/check-out | Check out |
| POST | /api/hospitality/pms/reservations/{id}/early-check-in | Early check-in |
| POST | /api/hospitality/pms/reservations/{id}/late-check-out | Late check-out |
| GET | /api/hospitality/pms/reservations/{id}/folio | Folio summary |
| POST | /api/hospitality/pms/reservations/{id}/folio/charges | Add charge |
| DELETE | /api/hospitality/pms/folio/{folio_id} | Void charge |
| POST | /api/hospitality/pms/reservations/{id}/payments | Post payment |
| GET | /api/hospitality/pms/housekeeping | List HK tasks |
| POST | /api/hospitality/pms/housekeeping | Create HK task |
| PUT | /api/hospitality/pms/housekeeping/{id} | Update HK task |
| POST | /api/hospitality/pms/housekeeping/{id}/complete | Complete HK task |
| DELETE | /api/hospitality/pms/housekeeping/{id} | Cancel HK task |
| GET | /api/hospitality/pms/group-bookings | List group bookings |
| POST | /api/hospitality/pms/group-bookings | Create group booking |
| POST | /api/hospitality/pms/group-bookings/{id}/confirm | Confirm group booking |
| POST | /api/hospitality/pms/night-audit | Run night audit |
| GET | /api/hospitality/pms/night-audit/{id} | Get audit report |
| GET | /api/hospitality/pms/dashboard | Dashboard summary |
| GET | /api/hospitality/pms/audit-events | Audit log |

### v2 Additions

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/pms/rooms/{id}/dynamic-rate | Dynamic rate for dates |
| PUT | /api/hospitality/pms/pricing-config | Update pricing config |
| GET | /api/hospitality/pms/analytics/revpar | RevPAR / ADR / TRevPAR |
| POST | /api/hospitality/pms/guests/{id}/loyalty/accrue | Accrue loyalty points |
| POST | /api/hospitality/pms/guests/{id}/loyalty/redeem | Redeem loyalty points |
| POST | /api/hospitality/pms/reservations/{id}/walk | Walk (relocate) guest |
| GET | /api/hospitality/pms/work-orders | List work orders |
| POST | /api/hospitality/pms/work-orders | Create work order |
| POST | /api/hospitality/pms/work-orders/{id}/assign | Assign work order |
| POST | /api/hospitality/pms/work-orders/{id}/start | Start work order |
| POST | /api/hospitality/pms/work-orders/{id}/close | Close work order |
| GET | /api/hospitality/pms/rate-plans | List rate plans |
| POST | /api/hospitality/pms/rate-plans | Create rate plan |
| GET | /api/hospitality/pms/rooms/{id}/applicable-rate | Best rate for room/dates |
| POST | /api/hospitality/pms/reservations/{id}/fiscal-receipt | Issue fiscal receipt |
| GET | /api/hospitality/pms/fiscal-receipts/{id} | Get fiscal receipt |
| GET | /api/hospitality/pms/fiscal-receipts | List fiscal receipts |
| POST | /api/hospitality/pms/housekeeping/auto-assign | Auto-assign HK tasks |

---

## Quick Usage Examples

### 1. Dynamic Pricing

Get a demand-adjusted rate before quoting a guest:

```
GET /api/hospitality/pms/rooms/{room_id}/dynamic-rate
    ?check_in=2026-07-10&check_out=2026-07-13
X-Tenant-ID: hotel_001

Response:
{
  "room_id": "...",
  "base_rate": "15000.00",
  "adjusted_rate": "18750.00",
  "multiplier": "1.2500",
  "demand_level": "high",
  "occupancy_ratio": 0.87,
  "nights": 3,
  "total_estimate": "56250.00"
}
```

Configure pricing bounds:

```
PUT /api/hospitality/pms/pricing-config
{
  "min_multiplier": "0.75",
  "max_multiplier": "2.00",
  "high_demand_threshold": 0.80,
  "low_demand_threshold": 0.35,
  "loyalty_earn_rate": "1.5"
}
```

### 2. Fiscal Receipt (VAT-compliant)

Issue a tamper-evident receipt after check-out:

```
POST /api/hospitality/pms/reservations/{res_id}/fiscal-receipt
X-Tenant-ID: hotel_001

{
  "receipt_lines": [
    {"description": "Room 101 — 3 nights", "quantity": 3, "unit_price": "15000.00"},
    {"description": "Minibar", "quantity": 1, "unit_price": "850.00"}
  ],
  "tax_rate": "0.16"
}

Response:
{
  "fiscal_sequence_number": 42,
  "subtotal": "45850.00",
  "tax_rate": "0.16",
  "tax_amount": "7336.00",
  "grand_total": "53186.00",
  "receipt_hash": "3a9f1c...",
  "currency": "KES"
}
```

### 3. Auto Housekeeping Assignment

Distribute all pending tasks across the morning shift:

```
POST /api/hospitality/pms/housekeeping/auto-assign
{
  "date": "2026-07-10",
  "staff_roster": [
    {"staff_id": "hk01", "name": "Alice", "section": "floors_1_3"},
    {"staff_id": "hk02", "name": "Bob",   "section": "floors_4_6"},
    {"staff_id": "hk03", "name": "Carol", "section": "floors_1_3"}
  ]
}

Response:
{
  "assigned_count": 12,
  "staff_loads": {"hk01": 5, "hk02": 4, "hk03": 3},
  "assignments": [...]
}
```

---

## Integration Notes

| APG Capability | Integration |
|----------------|-------------|
| `hos_fnb` (F&B) | Post restaurant and minibar charges directly to PMS folio via `add_folio_charge(charge_type="food")` |
| `hos_spa` | Spa treatment charges posted to folio; reservation ID used as the link key |
| `hos_channel` | OTA allotment and rate sync via `sync_channel_availability`; ingestion via `ingest_ota_reservation` (I3, planned) |
| `fin_accounts` | Night audit totals feed into accounts receivable; fiscal receipts exported to `fin_general_ledger` |
| `crm_loyalty` | Guest loyalty balance synced; `accrue_loyalty_points` / `redeem_loyalty_points` callable from CRM campaigns |
| `ops_maintenance` | Work orders can be escalated to `ops_maintenance` for property-wide asset tracking |
| `intel_reporting` | RevPAR analytics and night audit data surfaced in `intel_reporting` dashboards |
