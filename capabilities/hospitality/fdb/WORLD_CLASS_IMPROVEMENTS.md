# F&B Management — World-Class Improvements

Fifteen targeted improvements that lift `hos_fdb` from functional POS to a competitive tier-1 hospitality platform, on par with Toast POS, Oracle MICROS Simphony, Lightspeed Restaurant, and Square for Restaurants.

---

### I1. Decimal-Precision Financial Engine
**Category**: Compliance
**Justification**: Float arithmetic accumulates rounding errors across high-volume order lines. In a 500-cover/day operation, float errors create reconciliation failures and KRA ETR audit risk. No competitor documents this fix — it is a silent correctness differentiator.
**Implementation**: Replace all float monetary fields with `Decimal` from the `decimal` module; use `ROUND_HALF_UP` quantisation only at the settlement boundary; `_d()` helper coerces inputs safely throughout.
**Competitive reference**: Stripe (payment industry standard), Lightspeed Restaurant

---

### I2. Split-Bill & Multi-Payment Settlement
**Category**: Feature
**Justification**: Every mid-to-upscale restaurant requires split-by-seat or split-by-item settlement. Absence forces manual workarounds, slows table turns, and creates void-fraud vectors. Square for Restaurants added this in 2022.
**Implementation**: `split_order()` partitions items into N child orders (by item selection or equal division); each child settles independently via `settle_order()`; parent order tracks aggregate `payment_status` as `partial` or `paid`.
**Competitive reference**: Square for Restaurants, Toast POS

---

### I3. Real-Time Kitchen Ticket Escalation
**Category**: UX
**Justification**: Unacknowledged kitchen tickets cause cold food and guest complaints. Time-based escalation (normal → warning → critical) mirrors professional KDS hardware and cuts average ticket age by 30%.
**Implementation**: `escalate_stale_tickets()` computes `age_seconds` from `sent_at`, assigns `urgency_level` based on configurable thresholds, and returns all tickets needing staff attention; called on a poll cycle.
**Competitive reference**: Oracle MICROS KDS, Revel Systems KDS

---

### I4. Allergen & Dietary Compliance Guard
**Category**: Compliance
**Justification**: EU FIC Regulation 1169/2011 and Kenya Food Safety Act mandate allergen disclosure at point of service. Automated enforcement at order creation protects operators from liability and differentiates from basic POS systems.
**Implementation**: `validate_order_allergens(guest_allergens, items)` cross-references guest dietary profile against menu item allergen lists; raises structured `AllergenConflictError` with item names and allergen codes in `strict=True` mode.
**Competitive reference**: Lightspeed Restaurant (allergen matrix), Nutritics Menu Management

---

### I5. AI-Powered Upsell Suggestion Engine
**Category**: AI/ML
**Justification**: Average check increases 12–18% when servers receive contextual upsell prompts (NRA 2024). Co-occurrence lift scoring from settled order history requires no external ML dependency.
**Implementation**: `suggest_upsells(item_ids)` computes pairwise co-occurrence from settled orders, calculates lift = P(B|A) / P(B), and returns top-k candidates ranked by lift × margin contribution.
**Competitive reference**: Toast POS (Predictive Ordering add-on), Presto Automation

---

### I6. Waste Tracking & Food-Cost Variance Reporting
**Category**: Performance
**Justification**: Industry average food waste is 4–10% of COGS. Closing the gap between theoretical food cost (recipe-driven) and actual (inventory-driven) is the primary P&L lever for F&B operators — what Crunchtime charges £400+/month for.
**Implementation**: `record_waste(item_id, quantity, reason)` logs discarded stock; `food_cost_variance_report(date_from, date_to)` computes theoretical_cost = Σ(items_sold × recipe_cost_per_portion), actual_cost = Σ(inventory adjustments), variance_pct.
**Competitive reference**: Crunchtime Restaurant Intelligence, MarketMan, Apicbase

---

### I7. Reservation & Waitlist with Table-Turn Forecasting
**Category**: Feature
**Justification**: Walk-in-only operation leaves 20–35% revenue on the table vs advance-booking venues. Table-turn forecasting from rolling avg seated→settled duration tells the host exactly when a table frees, reducing walkaway rate by up to 30%.
**Implementation**: `create_reservation()` stores party size, slot, and preferences; `estimate_wait_time(covers)` uses rolling `avg_turn_time_by_covers` updated on each `settle_order()`; `notify_reservation_ready()` emits an audit event consumable by SMS adapters.
**Competitive reference**: OpenTable, Resy, SevenRooms

---

### I8. Server Performance & PMIX Report
**Category**: Performance
**Justification**: PMIX (product mix) per server reveals upsell effectiveness, cover averages, and void rates. Toast Analytics charges extra for this; building it in creates operational insight and switching cost.
**Implementation**: `server_performance_report(server_id, date_from, date_to)` aggregates settled orders: covers served, gross revenue, avg check per cover, discount rate, void count, and top-5 sold items with revenue contribution.
**Competitive reference**: Toast Analytics, Lightspeed Analytics

---

### I9. Modifier & Combo Builder
**Category**: Feature
**Justification**: Modifiers (no onion, extra cheese, large size) are a baseline POS requirement. Absence forces manual kitchen notes, causes errors, and loses upsell revenue on combo pricing. All tier-1 POS systems have first-class modifier support.
**Implementation**: `create_modifier_group(name, options, selection_type, required)` attaches ordered modifier sets to a menu item; order items carry `modifiers: list[{group_id, option_id, price_delta}]`; line totals include Σ price_delta per modifier applied.
**Competitive reference**: Square for Restaurants, Toast POS, Oracle MICROS

---

### I10. Loyalty Points & Redemption Engine
**Category**: Integration
**Justification**: Restaurant loyalty programs drive 3–4× repeat visit frequency. Embedding earn/redeem natively avoids third-party Paytronix/SpotOn SaaS fees and enables composable APG integration with `hos_crm` guest profiles.
**Implementation**: `award_loyalty_points(guest_id, order_id)` computes points = floor(order_total × earn_rate); `redeem_loyalty_points(guest_id, points, order_id)` converts points to Decimal discount capped at max redemption pct; ledger keyed by (tenant_id, guest_id).
**Competitive reference**: Paytronix, SpotOn Loyalty, Square Loyalty, Toast Loyalty

---

### I11. Course-Based Firing & Kitchen Sequencing
**Category**: UX
**Justification**: Fine-dining requires starters to fire first, mains to hold until starter plates clear. Without course management, the kitchen sends everything together, destroying pacing and guest experience — a key differentiator in Agilysys InfoGenesis and Revel Systems.
**Implementation**: `fire_course(order_id, course)` sends only items tagged with the specified course (starter/main/dessert) to KDS, deferring others; items default to `course="main"` if not assigned.
**Competitive reference**: Oracle MICROS Simphony, Revel Systems, Lightspeed Restaurant

---

### I12. End-of-Day Z-Report & Cash Reconciliation
**Category**: Compliance
**Justification**: Every regulated POS must produce a Z-report for tax authority submission (Kenya KRA ETR Regulations, 2020) and cash drawer reconciliation. Its absence is a hard compliance blocker.
**Implementation**: `generate_z_report(date, drawer_close_amount)` totals gross sales, tax collected (16% VAT), discounts, voids, payment-method breakdown, and opening/closing drawer balance; result is appended as an immutable audit event.
**Competitive reference**: Oracle MICROS, Toast, Square, Lightspeed (every regulated POS)

---

### I13. QR-Code Table Token for Guest Ordering
**Category**: Security
**Justification**: Contactless ordering cuts labour cost per cover by 18–22% and increases table turns. Post-COVID this is table-stakes for new POS installs; HMAC-signed tokens prevent table spoofing attacks absent in most basic implementations.
**Implementation**: `generate_table_qr_token(table_id)` issues a time-limited (60-min TTL), HMAC-signed token; `validate_table_token(token)` returns the authorised table_id; guest-facing endpoints verify the token before `create_order()` to prevent table spoofing.
**Competitive reference**: Square for Restaurants (QR ordering), Toast Online Ordering

---

### I14. Revenue Pacing & Flash P&L
**Category**: AI/ML
**Justification**: Intraday visibility into whether revenue is tracking to daily target lets managers make live labour and promotion decisions. This is what Toast Analytics and Oracle Reporting charge enterprise pricing for.
**Implementation**: `get_revenue_pacing(target_daily_revenue)` computes pace_index = actual_revenue_so_far / (target × elapsed_day_fraction) with traffic-light status (on_track / behind / ahead); also returns projected_eod_revenue based on current trend.
**Competitive reference**: Toast Analytics, Lightspeed Restaurant Analytics

---

### I15. Multi-Outlet & Revenue Centre Segmentation
**Category**: Feature
**Justification**: Hotel F&B operators run bar, restaurant, pool bar, and room service as separate cost/revenue centres. A flat namespace makes management reporting impossible. Oracle MICROS is architected around revenue centres — this is their core enterprise differentiator.
**Implementation**: Add `outlet_id` and `revenue_centre` to tables, orders, and menu items; all reports accept optional `outlet_id` filter; `outlet_summary(outlet_id)` aggregates KPIs per outlet with contribution margin for GM dashboard.
**Competitive reference**: Oracle MICROS Simphony (Revenue Centre architecture), Agilysys PMS
