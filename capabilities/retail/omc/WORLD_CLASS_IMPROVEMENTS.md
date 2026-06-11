# World-Class Improvements: Omnichannel Commerce (retail_omc)

## Overview

Fifteen targeted improvements to elevate `retail_omc` from a functional service layer to production-grade omnichannel commerce infrastructure. Each improvement includes rationale, expected impact, and implementation guidance.

---

## 1. Distributed Inventory Reservation with TTL and Saga Pattern

**Current state**: `reserve_inventory` mutates in-memory state with no expiry. Crashed processes leak reservations permanently.

**Improvement**: Introduce a `ReservationLedger` that assigns each reservation a UUID, expiry timestamp, and saga correlation ID. A background sweep releases expired reservations and emits `inventory_reservation_expired` events. Pair with a two-phase commit saga across inventory and order creation so partial failures roll back atomically.

**Impact**: Eliminates oversell from TTL leaks; enables safe retries during checkout failures.

---

## 2. Real-Time Cross-Channel Inventory Broadcast via WebSocket / SSE

**Current state**: Inventory checks are point-in-time pull operations.

**Improvement**: Wrap `upsert_inventory` and `reserve_inventory` to publish delta events to an in-process `asyncio.Queue` per tenant. Expose an SSE endpoint `/retail-omc/api/v1/inventory/stream` that subscribes to the queue. Store UIs and fulfilment terminals receive sub-second stock updates without polling.

**Impact**: Eliminates "item in cart, out of stock at checkout" UX failures; enables live shelf-edge displays.

---

## 3. Multi-Touch Attribution Engine (Linear, Time-Decay, Data-Driven)

**Current state**: `channel_attribution` implements last-touch only.

**Improvement**: Add configurable attribution models — linear (equal credit), time-decay (exponential half-life configurable per tenant), and a data-driven model that uses historical conversion rates per channel pair. Store attribution vectors, not just winning channel. Feed results to analytics aggregator for channel ROI reporting.

**Impact**: Eliminates systematic under-investment in top-of-funnel channels that last-touch attribution punishes.

---

## 4. Predictive BOPIS Slot Management

**Current state**: `bopis_order` accepts any pickup date without capacity awareness.

**Improvement**: Model each store's pick capacity as a `SlotGrid` keyed by `(store_id, date, hour)`. On BOPIS creation, check slot availability, reserve a slot, and return an `estimated_ready_by` window. Expose slot calendars via API so the web channel renders accurate pickup windows. Auto-release slots on order cancellation.

**Impact**: Eliminates store pick queue overload; reduces BOPIS SLA breaches.

---

## 5. Order Splitting and Partial Fulfilment

**Current state**: `order_routing` computes routing decisions but does not split the order record.

**Improvement**: When `split_allowed=True` and multiple fulfilment nodes are required, materialize child shipment records linked to the parent order. Each shipment tracks its own status, carrier tracking, and item subset. Roll up shipment statuses to parent order status. Expose `list_shipments(order_id)` and `mark_shipment_shipped()`.

**Impact**: Enables accurate per-item tracking; prerequisite for partial refunds and partial returns.

---

## 6. Loyalty Points Composability Hook

**Current state**: No earn/burn lifecycle integration.

**Improvement**: Add `earn_loyalty_points(order_id, program_id)` and `burn_loyalty_points(cart_id, points, program_id)` methods that call out to `retail_loy` via the capability adapter pattern. The burn method applies a line-item discount and records promotion_id on the cart line. Emit `loyalty_earned` / `loyalty_burned` events.

**Impact**: Enables seamless points-at-checkout without tight coupling to the loyalty service.

---

## 7. Fraud Scoring Integration with Inline Decisioning

**Current state**: `OmcOrderResponse.fraud_check_passed` is hardcoded `False`; no scoring logic exists.

**Improvement**: Add `async def fraud_screen_order(order_id: str) -> dict[str, Any]` that assembles a feature vector (velocity, address risk, device fingerprint, value) and calls a configurable scoring adapter. Orders above configurable threshold auto-hold; high-risk orders trigger a manual review workflow via `wflo`. Store score, signals, and decision on the order record.

**Impact**: Reduces chargeback rate; satisfies PCI DSS requirement 10.2.

---

## 8. Dynamic Shipping Rate Calculation

**Current state**: `OmcOrderResponse.shipping_total` is always `0.0`.

**Improvement**: Add `async def calculate_shipping(order_id: str, carrier_options: list[str]) -> list[dict[str, Any]]` that evaluates carrier rate tables against order weight, dimensions, destination zone, and fulfilment mode. Return ranked carrier options with ETA and price. Store selected option on the order. Integrate with `schd` for next-day cut-off enforcement.

**Impact**: Surfaces accurate shipping costs at checkout, reducing cart abandonment from surprise shipping fees.

---

## 9. Cart Merge on Authentication (Guest-to-Registered)

**Current state**: Guest carts and authenticated carts are independent; no merge path exists.

**Improvement**: Add `async def merge_carts(tenant_id: str, guest_cart_id: str, authenticated_cart_id: str, strategy: str) -> OmcCartResponse` with strategies `keep_authenticated`, `keep_guest`, and `union` (merge unique SKUs, summing quantities where duplicated). Emit `cart_merged` journey event. Invalidate the source cart.

**Impact**: Core UX requirement for any retailer with guest checkout; eliminates lost carts on login.

---

## 10. Promotion and Coupon Engine with Stacking Rules

**Current state**: Coupon codes exist on `OmcOrderCreate` but are never applied.

**Improvement**: Add `async def apply_promotions(cart_id: str, coupon_codes: list[str]) -> OmcCartResponse` that resolves promotions from `retail_prm`, validates stacking rules (exclusive, additive, best-deal), and applies qualifying discounts as line-item records with `promotion_id`. Emit `promotion_applied` event. Track redemption count against promotion caps.

**Impact**: Enables marketing team to run targeted campaigns without engineering involvement.

---

## 11. Inventory Aggregation and Safety Stock Automation

**Current state**: `safety_stock_qty` is stored but never computed or enforced.

**Improvement**: Add `async def compute_safety_stock(sku: str, location_id: str, days: int) -> dict[str, Any]` that calculates reorder point from historical demand standard deviation and supplier lead time. Auto-update `safety_stock_qty` on the inventory record. Add `async def list_low_stock_alerts(threshold_multiplier: float) -> list[dict[str, Any]]` for purchase order triggers.

**Impact**: Prevents stockouts from demand spikes; reduces excess inventory carrying costs.

---

## 12. Structured Audit Trail with Event Sourcing

**Current state**: Logging is `logger.info` calls; no queryable audit log.

**Improvement**: Add `async def _emit_audit_event(entity_type: str, entity_id: str, action: str, before: dict, after: dict, actor: str)` that appends to `self._audit_log: list[dict]`. Expose `async def query_audit_log(entity_type: str, entity_id: str) -> list[dict]`. On recovery, state can be replayed from the audit log. Structured as CloudEvents for Bytewax compatibility.

**Impact**: Required for financial compliance; enables forensic investigation of inventory discrepancies.

---

## 13. Channel Health Monitoring and Circuit Breaker

**Current state**: Channels can be `is_active=False` but no automated health tracking exists.

**Improvement**: Track per-channel error rates, latency percentiles, and order failure counts in a sliding window. Add `async def get_channel_health(tenant_id: str, channel_id: str) -> dict[str, Any]` returning status, p95 latency, and error rate. Implement circuit breaker: auto-disable channels exceeding configurable thresholds and emit `channel_degraded` alert to `moni`.

**Impact**: Prevents cascading failures from a degraded channel polluting omnichannel metrics.

---

## 14. Catalogue Search with Faceted Filtering and Relevance Ranking

**Current state**: `list_catalogue_items` supports only exact `category_path` prefix matching.

**Improvement**: Add `async def search_catalogue(tenant_id: str, query: str, filters: dict[str, Any], sort: str, page: int, page_size: int) -> dict[str, Any]` with full-text matching on name/description, facet aggregation on brand/category/price range, and relevance scoring via BM25. Integrate `nlpc` capability for NLP-enhanced search when available.

**Impact**: Improves product discovery conversion; eliminates manual category browsing as primary navigation.

---

## 15. Return Merchandise Authorization (RMA) Workflow with Condition Grading

**Current state**: Returns are initiated and approved but have no physical processing workflow.

**Improvement**: Add `async def process_rma(return_id: str, received_items: list[dict], condition_grades: dict[str, str]) -> dict[str, Any]` that grades each returned item (`new`, `refurbished`, `damaged`, `scrap`), routes to restock, refurbishment queue, or write-off, and adjusts inventory accordingly. Compute actual refund from condition grades. Emit `rma_processed` event with disposition per item.

**Impact**: Closes the physical returns loop; enables resale of returned goods, directly improving recovery margin.
