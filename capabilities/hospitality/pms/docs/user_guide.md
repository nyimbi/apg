# PMS User Guide

## Overview

The Property Management System (`hos_pms`) manages the complete guest lifecycle — from rate enquiry through check-out and fiscal receipt issuance — for hospitality properties running on the APG platform. Version 2.0 adds demand-aware pricing, yield analytics, loyalty points, walk management, engineering work orders, a rate plan catalogue, VAT-compliant fiscal receipts, and automated housekeeping assignment.

---

## Architecture

All state is held in the `PMSService` class, instantiated per tenant. Each method accepts an optional `tenant_id` parameter; if omitted, `PMSService.tenant_id` is used. All monetary values are computed with `Decimal` internally; floats appear only in storage dict values for JSON compatibility.

```
PMSService
  ├── rooms               — room inventory
  ├── guests              — guest profiles + loyalty balance
  ├── reservations        — stay records with folio balance
  ├── folios              — line-item charges
  ├── payments            — payment records
  ├── housekeeping_tasks  — HK task queue
  ├── work_orders         — engineering work orders (I7)
  ├── rate_plans          — rate plan catalogue (I9)
  ├── fiscal_receipts     — write-once fiscal records (I12)
  ├── group_bookings      — group/block bookings
  └── night_audits        — end-of-day audit snapshots
```

---

## Key Use Cases

### Room Inventory

Register every room with its type, floor, capacity, nightly rate, and amenities. Room status transitions automatically as guests move through the lifecycle:

```
available → occupied (check-in)
occupied  → housekeeping (check-out)
housekeeping → available (HK task completed)
available → maintenance (set_room_status)
```

### Guest Profiles

Profiles store contact details, government ID, VIP tier, stay count, total spend, and (v2) loyalty point balance. Deactivating a guest soft-deletes the record.

### Reservations

Create a reservation by supplying `guest_id`, `room_id`, and dates. The system validates no overlapping confirmed/checked-in reservations exist for that room. `total_amount` and `balance` are recalculated whenever dates or room change.

Reservation status flow:
```
confirmed → checked_in → checked_out
confirmed → cancelled
confirmed/checked_in → walked  (I6)
```

### Check-In / Check-Out

`check_in` transitions the reservation to `checked_in` and marks the room `occupied`. It increments `guest.stay_count`.

`check_out` settles any final payment, marks the room `housekeeping`, and adds the stay total to `guest.total_spend`.

### Folio Management

All charges are line items in `folios`. Charge types: `room`, `food`, `beverage`, `spa`, `laundry`, `telephone`, `loyalty_redemption`, `walk_compensation`, `other`. Each charge updates `reservation.total_amount` and `reservation.balance`. Void a charge with `void_folio_charge` to reverse the balance.

### Housekeeping

Create tasks with `create_housekeeping_task(task_type=...)`. Types: `clean`, `turndown`, `inspect`, `maintenance`. Completing a `clean`, `turndown`, or `inspect` task on a `housekeeping` room automatically sets the room back to `available`.

Use `auto_assign_housekeeping` (I13) to distribute all pending tasks across a shift roster automatically.

### Night Audit

`run_night_audit(audit_date, run_by)` produces an immutable snapshot of:
- Total and occupied rooms, occupancy rate
- Arrivals, departures, no-shows, walk-ins
- Room revenue and ancillary revenue from all in-house folios

Run it once per day after midnight.

### Group Bookings

Create a block with `create_group_booking` (status: `tentative`). Confirm with `confirm_group_booking`. Individual reservations under the block reference the group ID.

---

## v2 Feature Guides

### I1 — Dynamic Pricing Engine

The engine adjusts the nightly rate based on current occupancy and booking lead time. It linearly interpolates a multiplier between configurable bounds.

**Configure bounds** (once per property, or seasonally):

```python
await svc.configure_pricing(
    min_multiplier="0.80",       # floor at 80 % of base rate
    max_multiplier="1.50",       # ceiling at 150 %
    high_demand_threshold=0.85,  # occupancy at which ceiling applies
    low_demand_threshold=0.40,   # occupancy at which floor applies
    loyalty_earn_rate="1.5",     # points earned per KES spent
)
```

**Get a dynamic rate** before quoting:

```python
rate = await svc.get_dynamic_rate(room_id, "2026-08-01", "2026-08-04")
# rate["adjusted_rate"] is the quoted price
# rate["demand_level"] is "low" | "moderate" | "high"
```

Bookings with more than 30 days of lead time receive a 5-percentage-point discount on the computed multiplier.

---

### I2 — RevPAR / Yield Analytics

Compute hotel yield KPIs for any date range:

```python
analytics = await svc.get_revpar_analytics("2026-07-01", "2026-07-31")
# analytics["revpar"]   — Revenue Per Available Room
# analytics["adr"]      — Average Daily Rate
# analytics["trevpar"]  — Total Revenue Per Available Room (includes ancillary)
# analytics["occupancy_pct"]
```

All values are Decimal strings to preserve precision in downstream accounting integrations.

---

### I5 — Loyalty Points

Points accrue at `loyalty_earn_rate` per base currency unit (default: 1 pt / KES 1).

**Accrue** after check-out:

```python
result = await svc.accrue_loyalty_points(guest_id, reservation_id, spend_amount="45850.00")
# result["points_earned"], result["new_balance"]
```

**Redeem** at check-in or any time during a stay:

```python
result = await svc.redeem_loyalty_points(guest_id, points=5000, reservation_id=reservation_id)
# Posts a KES -5000 folio credit automatically
```

---

### I6 — Overbooking Walk Management

When a property is oversold and a guest must be relocated:

```python
walk = await svc.walk_reservation(
    reservation_id=res_id,
    relocation_property="Acacia Hotel Nairobi",
    reason="Oversell — system error",
    covered_costs="15000.00",   # transport + first night at relocation property
    walked_by="front_desk_manager",
)
# walk["comp_charge_id"] — folio credit ID for the covered costs
```

The reservation status becomes `walked`, the room is freed to `available`, and a compensation credit is posted to the folio. All walk events are logged at WARNING level for duty-of-care audit purposes.

---

### I7 — Maintenance Work Orders

Engineering issues follow a four-stage lifecycle with SLA enforcement:

```
reported → assigned → in_progress → closed
```

SLA thresholds by priority:

| Priority | SLA |
|----------|-----|
| urgent   | 2 h |
| high     | 8 h |
| normal   | 24 h |
| low      | 72 h |

```python
wo = await svc.create_work_order(
    room_id=room_id,
    category="plumbing",
    description="Shower drain blocked — room 205",
    priority="high",
    reported_by="housekeeper_01",
)
await svc.assign_work_order(wo["id"], assigned_to="engineer_02")
await svc.start_work_order(wo["id"])
closed = await svc.close_work_order(wo["id"], verified_by="chief_engineer")
# closed["sla_breach"] is True if resolution exceeded 8 h
```

Closing a work order on a room in `maintenance` status returns the room to `available`.

---

### I9 — Rate Plan Catalogue

Define named rate plans with restriction rules:

```python
plan = await svc.create_rate_plan(
    code="EARLYBIRD30",
    name="Early Bird — 30 days advance",
    base_rate="12000.00",
    min_stay=2,
    advance_purchase_days=30,
    applicable_room_types=["deluxe", "suite"],
)
```

At booking time, retrieve the best available plan:

```python
rate = await svc.get_applicable_rate(room_id, "2026-09-01", "2026-09-04")
# rate["rate_plan_code"], rate["nightly_rate"], rate["source"]
# source is "rate_plan" or "room_base" (fallback)
```

The system evaluates all active plans and returns the lowest qualifying rate.

---

### I12 — Fiscal Compliance Receipts

Issue a tamper-evident VAT receipt after settlement:

```python
receipt = await svc.post_fiscal_receipt(
    reservation_id=res_id,
    receipt_lines=[
        {"description": "Room 101 — 3 nights", "quantity": 3, "unit_price": "15000.00"},
        {"description": "Restaurant charges",   "quantity": 1, "unit_price": "3200.00"},
    ],
    tax_rate="0.16",   # 16 % VAT
)
# receipt["fiscal_sequence_number"] — monotonically incrementing
# receipt["receipt_hash"] — SHA-256 chain hash
# receipt["grand_total"]  — KES inclusive of VAT
```

Each receipt is stored in a write-once dict. The chain hash links each receipt to its predecessor, making retroactive tampering detectable. Receipts are suitable for submission to the Kenya Revenue Authority (KRA) iTax portal.

---

### I13 — Automated Housekeeping Assignment

Distribute all pending tasks across the morning shift in one call:

```python
result = await svc.auto_assign_housekeeping(
    date="2026-07-10",
    staff_roster=[
        {"staff_id": "hk01", "name": "Alice", "section": "floors_1_3"},
        {"staff_id": "hk02", "name": "Bob",   "section": "floors_4_6"},
    ],
)
# result["assigned_count"] — total tasks assigned
# result["staff_loads"]    — {"hk01": 7, "hk02": 5}
# result["assignments"]    — full task list with staff allocation
```

Task scoring: checkout rooms (3 pts) > stayover turndown/inspect (2 pts) > other (1 pt). Staff are distributed by floor section to reduce corridor travel. Load is balanced within each section.

---

## Error Reference

| Exception | Meaning |
|-----------|---------|
| `KeyError: room_not_found:X` | Room ID not found for this tenant |
| `KeyError: guest_not_found:X` | Guest ID not found for this tenant |
| `KeyError: reservation_not_found:X` | Reservation not found |
| `ValueError: cannot_check_in_from_status:X` | Reservation is not in `confirmed` state |
| `ValueError: cannot_walk_reservation_in_status:X` | Walk attempted on closed reservation |
| `ValueError: insufficient_loyalty_balance:N` | Guest has fewer points than requested redemption |
| `ValueError: sla_breach` | Work order closed after SLA; flagged in record, not raised |
| `PermissionError: tenant_context_required` | No tenant ID supplied or configured |

---

## Integration with Other APG Capabilities

| Capability | How to connect |
|------------|----------------|
| `hos_fnb` | Call `pms.add_folio_charge(charge_type="food")` from the F&B service after an order is settled |
| `hos_spa` | Same pattern with `charge_type="spa"` |
| `fin_accounts` | Export `run_night_audit` output nightly to `fin_accounts.post_journal_entry` |
| `fin_general_ledger` | Map `fiscal_receipts` to GL by charge type using the `fiscal_sequence_number` as the external reference |
| `crm_loyalty` | Sync `guest.loyalty_balance` bidirectionally; loyalty campaigns call `accrue_loyalty_points` on qualifying stays |
| `intel_reporting` | Surface `get_revpar_analytics` and `dashboard_summary` output in reporting dashboards |
| `ops_maintenance` | Escalate work orders to `ops_maintenance` for multi-property asset tracking |
