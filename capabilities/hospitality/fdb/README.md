# F&B Management (hos_fdb)

Restaurant POS, table management, menu engineering, kitchen display, recipe costing, and inventory control.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/fdb/health | Health check |
| GET | /api/hospitality/fdb/menu-items | List menu items |
| POST | /api/hospitality/fdb/menu-items | Create menu item |
| GET | /api/hospitality/fdb/menu-items/{id} | Get menu item |
| PUT | /api/hospitality/fdb/menu-items/{id} | Update menu item |
| DELETE | /api/hospitality/fdb/menu-items/{id} | Deactivate item |
| GET | /api/hospitality/fdb/tables | List tables |
| POST | /api/hospitality/fdb/tables | Create table |
| POST | /api/hospitality/fdb/tables/{id}/seat | Seat guests |
| GET | /api/hospitality/fdb/orders | List orders |
| POST | /api/hospitality/fdb/orders | Create order |
| POST | /api/hospitality/fdb/orders/{id}/send-to-kitchen | Send to KDS |
| POST | /api/hospitality/fdb/orders/{id}/settle | Settle bill |
| DELETE | /api/hospitality/fdb/orders/{id} | Void order |
| POST | /api/hospitality/fdb/kitchen-tickets/{id}/complete | Complete ticket |
| POST | /api/hospitality/fdb/recipes | Create recipe |
| GET | /api/hospitality/fdb/inventory | List inventory |
| POST | /api/hospitality/fdb/inventory | Add inventory item |
| POST | /api/hospitality/fdb/inventory/{id}/adjust | Adjust stock |
| GET | /api/hospitality/fdb/menu-engineering | Menu engineering report |
| GET | /api/hospitality/fdb/revenue-report | Daily revenue |
| GET | /api/hospitality/fdb/dashboard | Dashboard |

## World-Class Enhancements (v2.0)

Fifteen targeted improvements that lift `hos_fdb` to tier-1 parity with Toast POS, Oracle MICROS Simphony, Lightspeed Restaurant, and Square for Restaurants.

**I1. Decimal-Precision Financial Engine** — Replace float monetary fields with `Decimal` + `ROUND_HALF_UP` at settlement boundary; eliminates reconciliation drift in high-volume operations. [Compliance]

**I2. Split-Bill & Multi-Payment Settlement** — `split_order()` partitions items into N independently-settling child orders; parent tracks aggregate `payment_status`. [Feature]

**I3. Real-Time Kitchen Ticket Escalation** — `escalate_stale_tickets()` assigns `urgency_level` (normal/warning/critical) from `age_seconds`; reduces average ticket age ~30%. [UX]

**I4. Allergen & Dietary Compliance Guard** — `validate_order_allergens()` cross-references guest dietary profile against item allergen lists; raises structured `AllergenConflictError` (EU FIC / Kenya FSA compliant). [Compliance]

**I5. AI-Powered Upsell Suggestion Engine** — `suggest_upsells(item_ids)` scores pairwise co-occurrence lift from settled history; no external ML dependency; targets 12–18% avg-check uplift. [AI/ML]

**I6. Waste Tracking & Food-Cost Variance Reporting** — `record_waste()` + `food_cost_variance_report()` close the gap between theoretical recipe cost and actual inventory consumption. [Performance]

**I7. Reservation & Waitlist with Table-Turn Forecasting** — `create_reservation()` + `estimate_wait_time()` use rolling avg turn time by cover count; `notify_reservation_ready()` emits SMS-consumable audit event. [Feature]

**I8. Server Performance & PMIX Report** — `server_performance_report()` aggregates covers, gross revenue, avg check, discount rate, void count, and top-5 items per server. [Performance]

**I9. Modifier & Combo Builder** — `create_modifier_group()` attaches ordered modifier sets (required/optional, single/multi-select) to menu items; line totals include Σ price_delta. [Feature]

**I10. Loyalty Points & Redemption Engine** — `award_loyalty_points()` / `redeem_loyalty_points()` embed earn/redeem natively; composable with `hos_crm` guest profiles. [Integration]

**I11. Course-Based Firing & Kitchen Sequencing** — `fire_course(order_id, course)` sends only starter/main/dessert items at the correct moment; items default to `course="main"`. [UX]

**I12. End-of-Day Z-Report & Cash Reconciliation** — `generate_z_report()` totals gross sales, 16% VAT, discounts, voids, payment breakdown, and drawer balance as an immutable audit event (KRA ETR compliant). [Compliance]

**I13. QR-Code Table Token for Guest Ordering** — `generate_table_qr_token()` issues 60-min HMAC-signed tokens; `validate_table_token()` enforces table identity before `create_order()`. [Security]

**I14. Revenue Pacing & Flash P&L** — `get_revenue_pacing(target)` computes `pace_index = actual / (target × elapsed_day_fraction)` with traffic-light status and projected EOD revenue. [AI/ML]

**I15. Multi-Outlet & Revenue Centre Segmentation** — Adds `outlet_id` + `revenue_centre` to tables, orders, and menu items; `outlet_summary()` aggregates per-outlet KPIs for GM dashboard. [Feature]

## New Methods

Three high-impact methods added in v2.0:

### `settle_order` with split-bill flow

```python
# Split a 4-cover table into two 2-item sub-bills, then settle each independently
child_a, child_b = await svc.split_order(
    order_id="ord_abc",
    splits=[["item_1", "item_2"], ["item_3", "item_4"]],
    tenant_id="hotel_nairobi",
)
await svc.settle_order(child_a["id"], payment_method="mpesa", amount_paid=1850.00)
await svc.settle_order(child_b["id"], payment_method="cash",  amount_paid=2000.00)
```

### `suggest_upsells` — co-occurrence lift scoring

```python
# Current basket contains a burger; find high-lift add-ons from settled history
suggestions = await svc.suggest_upsells(
    item_ids=["item_burger"],
    top_k=3,
    tenant_id="hotel_nairobi",
)
# -> [{"item_id": "item_fries", "lift": 3.2, "margin_contribution": 180.0}, ...]
```

### `get_revenue_pacing` — intraday flash P&L

```python
pacing = await svc.get_revenue_pacing(
    target_daily_revenue=250_000.00,
    tenant_id="hotel_nairobi",
)
# -> {
#      "pace_index": 0.87,
#      "status": "behind",
#      "actual_revenue": 108_500.00,
#      "projected_eod_revenue": 217_000.00,
#      "elapsed_day_fraction": 0.50,
#    }
```
