# World-Class Improvements — Order Management & Tracking (scm_omt)

## 1. Order Line-Level Status Tracking

Currently, order lines carry a single `status: "draft"` field set at creation and never updated.
Individual lines should transition independently — a line can be `allocated`, `backordered`,
`picked`, `packed`, `shipped`, or `cancelled` while others in the same order are at a different
stage. This unlocks partial-fulfilment workflows and accurate split-shipment triggers.

## 2. Partial Fulfilment & Overship Guard

There is no guard preventing a split-shipment from shipping more quantity than was ordered, nor
any mechanism for recording that a line has been partially fulfilled. Add a
`shipped_quantity` accumulator per line and reject shipment requests that would exceed the
ordered quantity. Partial fulfilment should automatically promote the order to `partially_shipped`.

## 3. ATP Horizon Simulation (Date-Bucketed ATP)

The current ATP check is point-in-time; it has no notion of supply arriving on future dates.
Extend `update_atp` to accept a list of dated `supply_events` (PO receipts, production completions)
and `demand_events` (confirmed orders), then compute a rolling ATP profile so
`check_atp(sku, qty, requested_date)` returns whether stock will be available *by* the requested
date rather than right now. This is the foundation of capable-to-promise (CTP) logic.

## 4. Dynamic Re-promising Engine

When ATP drops or a supplier delivery is delayed, all active delivery promises that depended on
that stock become stale. A re-promising pass should scan active promises, compare promised dates
against the revised ATP profile, flag breached promises, and optionally auto-revoke them with a
system-generated reason. The customer notification pipeline should be triggered automatically.

## 5. Order Scoring & Priority Lanes

Customers have different SLA tiers (strategic accounts vs. standard), orders have different
revenue weights, and fulfilment capacity is constrained. Implement a composite score on each
order (`revenue × priority_weight × customer_tier_weight`) and expose a `get_order_queue`
method that returns confirmed orders sorted by score, giving the warehouse a priority-sorted
pick list rather than FIFO.

## 6. Rule-Based Order Routing

Split-shipments today require a human to decide which warehouse handles which line. Add a
`route_order` method that accepts a set of warehouse ATP snapshots and a routing policy (closest
warehouse first, cheapest freight first, consolidate-if-possible) and returns an assignment plan
mapping each line to a warehouse. This integrates with the WMS `scm_wms` capability.

## 7. Idempotency Keys on Order Creation

Without idempotency keys, a network retry can create duplicate orders. Accept an optional
`idempotency_key` on `create_order`; if a matching key already exists for the tenant, return
the original order instead of creating a new one. Store keys in a bounded LRU cache with a
configurable TTL (default 24 h).

## 8. Configurable Order State-Machine

The allowed status transitions are implicit and scattered across individual methods. Define a
formal adjacency map `TRANSITIONS: dict[str, set[str]]` and a single `_assert_transition`
guard. This prevents an order jumping from `draft` directly to `shipped` via a poorly validated
`update_order` call and makes the state machine auditable from code alone.

## 9. Bulk Operations with Concurrency Cap

`bulk_confirm_orders` uses unbounded `asyncio.gather`. For large batches this can spike DB
connections / rate-limit downstream services. Wrap it in a semaphore-controlled
`_bounded_gather` helper (default concurrency = 10) and apply the same pattern to any future
bulk mutation method.

## 10. Carrier Integration Adapter Interface

Shipping status (in-transit, out-for-delivery, delivered) currently has to be manually updated.
Define an abstract `CarrierAdapter` protocol with `fetch_tracking_events(tracking_number)`
returning normalized `TrackingEvent` objects, and a `sync_shipment_tracking` service method
that polls all `shipped` orders, pulls carrier events, and updates order status + customer
notifications automatically.

## 11. Tax & Duty Calculation Hook

Orders crossing international borders require tax/duty amounts before a binding promise can be
given to the customer. Add a `TaxEngine` protocol (compute_tax(lines, destination) →
TaxBreakdown) and invoke it in `confirm_order`, storing the result on the order record. Tax
engine implementations can be swapped via DI — Avalara, TaxJar, or a flat-rate fallback.

## 12. Customer Return & Reverse Logistics (RMA)

There is no return merchandise authorization flow. Add `create_rma`, `approve_rma`,
`receive_return`, and `process_refund` methods. An RMA should link back to its origin order,
capture condition codes (`new`, `damaged`, `missing_parts`), and trigger inventory adjustments
via the WMS capability event bus.

## 13. Delivery Window Negotiation

Rather than a single `requested_delivery_date`, customers often need to select from available
windows. Add a `get_available_delivery_windows` method that reads warehouse calendars (business
hours, cut-off times, blackout dates) and ATP horizon data to return a list of feasible delivery
date/time windows. The customer selects one, and `promise_order` is called with that commitment.

## 14. SLA Breach Detection & Escalation

Orders approaching or past their promised delivery date without reaching `delivered` status
should raise a breach event. Add `detect_sla_breaches` which computes
`now > promised_date AND status NOT IN {delivered, cancelled}`, emits a `sla_breach_detected`
audit event, and queues escalation notifications to account managers. Integrate with a
background scheduler (e.g. APScheduler or Celery beat).

## 15. Event-Sourced Audit Trail with Causality Chain

The current `_audit_events` list records what happened but not *why* — there is no causal link
between related events (e.g. which `atp_updated` event caused `backorder_created`). Add
`causation_id` (the triggering event id) and `correlation_id` (the root workflow id) to every
audit event. This enables full causal trace reconstruction for dispute resolution, regulatory
audits, and root-cause analysis of fulfilment failures.
