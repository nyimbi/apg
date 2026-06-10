# PMS User Guide

## Overview

The Property Management System (hos_pms) manages the full guest lifecycle from reservation through check-out, including room inventory, housekeeping workflows, folio charges, and nightly financial audits.

## Key Use Cases

- **Room Inventory**: Maintain a registry of all rooms with type, floor, capacity, rate, and amenities. Set status (available/occupied/maintenance/housekeeping/out_of_order).
- **Guest Profiles**: Store guest contact details, ID documents, VIP tiers, and stay history.
- **Reservations**: Create, modify, and cancel reservations with automatic balance computation.
- **Check-In / Check-Out**: Move reservations through the guest lifecycle; rooms update status automatically.
- **Folio Management**: Post room charges, F&B charges, spa, laundry, and miscellaneous charges. Void incorrect postings. Record payments.
- **Housekeeping**: Assign clean/turndown/inspect/maintenance tasks with priority and staff assignment.
- **Night Audit**: Run the end-of-day audit to capture occupancy, revenue, arrivals, departures, and no-shows.
- **Group Bookings**: Block multiple rooms under a single group with a contracted rate.

## API Reference

See README.md for endpoint table.

### Example: Create a Room

```
POST /api/hospitality/pms/rooms
X-Tenant-ID: hotel_001

{
  "room_number": "101",
  "room_type": "deluxe",
  "floor": 1,
  "capacity": 2,
  "rate_per_night": 15000.00,
  "amenities": ["wifi", "air_conditioning", "minibar"]
}
```

### Example: Make a Reservation

```
POST /api/hospitality/pms/reservations
{
  "guest_id": "<guest_id>",
  "room_id": "<room_id>",
  "check_in_date": "2026-07-01",
  "check_out_date": "2026-07-04",
  "adults": 2,
  "source": "direct"
}
```

### Example: Run Night Audit

```
POST /api/hospitality/pms/night-audit
{
  "audit_date": "2026-07-01",
  "run_by": "night_manager"
}
```
