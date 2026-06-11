# Omnichannel Commerce — User Guide

**Capability ID**: `retail_omc` | **Domain**: `retail` | **Version**: `1.1.0`

## Description

`retail_omc` provides unified commerce orchestration across all retail touchpoints: channel registry, cross-channel inventory visibility with reservation, unified cart and order management, buy-online-pickup-in-store (BOPIS/C&C), ship-from-store fulfilment, multi-channel returns with RMA grading, customer journey event tracking with multi-touch attribution, fraud screening, shipping rate calculation, loyalty composability hooks, safety stock automation, and a structured audit trail. All operations are tenant-isolated.

---

## Installation

```bash
pip install apg-retail-omc
```

---

## Quick Start

```python
import asyncio
from capabilities.retail.omc.service import OmcService
from capabilities.retail.omc.models import (
    OmcChannelCreate, OmcCatalogueItemCreate,
    OmcInventoryRecord, OmcOrderCreate, OmcCartLineItem,
)

async def main():
    svc = OmcService(tenant_id="acme", actor_id="ops-user")

    # 1. Register a channel
    channel = await svc.create_channel(OmcChannelCreate(
        tenant_id="acme", name="Web Store", channel_type="web",
        currency_code="KES", created_by="ops-user",
    ))

    # 2. Add a catalogue item
    item = await svc.create_catalogue_item(OmcCatalogueItemCreate(
        tenant_id="acme", sku="SKU-001", name="Running Shoe",
        base_price=4500.0, currency_code="KES",
        category_path=["footwear", "running"],
        brand="TrailX", weight_kg=0.6,
        created_by="ops-user",
    ))

    # 3. Load inventory at a store
    from capabilities.retail.omc.models import OmcInventoryRecord
    await svc.upsert_inventory(OmcInventoryRecord(
        tenant_id="acme", sku="SKU-001",
        location_id="store-nairobi-01", channel_id=channel.id,
        on_hand_qty=50, available_qty=50, updated_by="ops-user",
    ))

    # 4. Create a BOPIS order
    result = await svc.bopis_order(
        customer_id="cust-001", sku="SKU-001", quantity=2,
        pickup_store="store-nairobi-01", pickup_date="2026-06-15",
    )
    order_id = result["order"]["id"]

    # 5. Mark ready for collection
    await svc.click_and_collect_ready(order_id)

    print(f"Order {order_id} ready for pickup")

asyncio.run(main())
```

---

## Provides

| Service | Description |
|---|---|
| `omnichannel_order_management` | Unified order lifecycle across all channels |
| `inventory_visibility` | Real-time cross-location stock with reservation |
| `click_and_collect` | BOPIS fulfilment with store assignment and slot awareness |
| `customer_journey_orchestration` | Stage-by-stage journey event tracking |
| `unified_cart` | Channel-agnostic cart with promotion and loyalty application |
| `cross_channel_fulfilment` | Ship-from-store, C&C, and home delivery routing |
| `omnichannel_search` | Faceted catalogue search with BM25 relevance ranking |
| `return_management` | Cross-channel return initiation, RMA grading, and refund approval |
| `channel_pricing_engine` | Per-channel price overrides with arbitrage prevention |
| `multi_touch_attribution` | Linear, time-decay, first/last-touch attribution models |
| `fraud_screening` | Heuristic fraud scoring with auto-hold on threshold breach |
| `shipping_rate_engine` | Multi-carrier rate calculation with weight and zone pricing |
| `loyalty_composability` | Earn/burn lifecycle hooks into `retail_loy` |
| `safety_stock_management` | Demand-volatility-based safety stock with low-stock alerting |
| `audit_trail` | Structured CloudEvent audit log with entity-level queryability |
| `cart_merge` | Guest-to-authenticated cart merge with configurable strategies |

---

## Requires

| Capability | Reason |
|---|---|
| `auth` | Customer and operator authentication |
| `audl` | Order and inventory audit trail |
| `mten` | Tenant context isolation |
| `conf` | Channel and fulfilment configuration |
| `ntfy` | Order status and loyalty notifications |
| `wflo` | High-value / high-fraud order approval workflow |
| `mqeb` | Bytewax inventory sync stream |
| `moni` | Inventory and fulfilment SLA monitoring |
| `nlpc` | Search NLP and personalisation (optional enhancement) |
| `schd` | Inventory sync and shipping cut-off scheduling |

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed `RETAIL_OMC_`.

| Key | Default | Description |
|---|---|---|
| `inventory.reservation_ttl_seconds` | 900 | Inventory reservation timeout |
| `inventory.safety_stock_enabled` | true | Safety stock buffer active |
| `cart.abandonment_timeout_minutes` | 60 | Cart abandonment threshold |
| `cart.max_items` | 200 | Maximum items per cart |
| `orders.approval_required_above_value` | 50000 | High-value order approval threshold (KES) |
| `returns.return_window_days` | 30 | Maximum days post-purchase for returns |
| `payments.pci_compliant_required` | true | PCI compliance mandatory |
| `RETAIL_OMC_FRAUD_THRESHOLD` | 0.7 | Fraud score threshold for auto-hold (0–1) |

---

## Service Methods Reference

### Channels

```python
await svc.create_channel(data: OmcChannelCreate) -> OmcChannelResponse
await svc.get_channel(tenant_id, channel_id) -> OmcChannelResponse | None
await svc.list_channels(tenant_id) -> list[OmcChannelResponse]
```

### Catalogue

```python
await svc.create_catalogue_item(data: OmcCatalogueItemCreate) -> OmcCatalogueItemResponse
await svc.get_catalogue_item(tenant_id, item_id) -> OmcCatalogueItemResponse | None
await svc.get_catalogue_item_by_sku(tenant_id, sku) -> OmcCatalogueItemResponse | None
await svc.list_catalogue_items(tenant_id, category_path=None) -> list[OmcCatalogueItemResponse]
await svc.set_channel_price(tenant_id, item_id, channel_id, price) -> OmcCatalogueItemResponse | None
await svc.search_catalogue(tenant_id, query, filters, sort, page, page_size) -> dict
```

**search_catalogue filters**:

| Filter | Type | Description |
|---|---|---|
| `category_path` | `list[str]` | Exact prefix match on category hierarchy |
| `brand` | `str` | Case-insensitive brand filter |
| `min_price` | `float` | Minimum base price |
| `max_price` | `float` | Maximum base price |
| `channel_id` | `str` | Only items with a channel-specific price |
| `in_stock_only` | `bool` | Only items with available inventory |

**sort options**: `relevance` (default), `price_asc`, `price_desc`, `name_asc`

### Inventory

```python
await svc.upsert_inventory(data: OmcInventoryRecord) -> OmcInventoryResponse
await svc.get_inventory(tenant_id, sku, location_id=None) -> list[OmcInventoryResponse]
await svc.unified_inventory_check(sku, channels) -> dict
await svc.reserve_inventory(tenant_id, sku, location_id, channel_id, qty) -> bool
await svc.release_inventory(tenant_id, sku, location_id, channel_id, qty) -> bool
await svc.compute_safety_stock(sku, location_id, lookback_days, lead_time_days, service_level_z) -> dict
await svc.list_low_stock_alerts(threshold_multiplier=1.0) -> list[dict]
```

**compute_safety_stock** uses the classical formula `SS = Z × σ_d × √L` where σ_d is daily demand standard deviation and L is supplier lead time in days.

**list_low_stock_alerts** returns items where `available_qty ≤ safety_stock_qty × threshold_multiplier`. Set `threshold_multiplier=1.5` for early warnings at 150% of safety stock.

### Cart

```python
await svc.create_cart(data: OmcCartCreate) -> OmcCartResponse
await svc.get_cart(tenant_id, cart_id) -> OmcCartResponse | None
await svc.abandon_cart(tenant_id, cart_id) -> OmcCartResponse | None
await svc.merge_carts(tenant_id, guest_cart_id, authenticated_cart_id, strategy="union") -> OmcCartResponse | None
await svc.burn_loyalty_points(cart_id, points, program_id, point_value=0.01) -> OmcCartResponse | None
```

**merge_carts strategies**:
- `union` — merge all SKUs; sum quantities for duplicates (default)
- `keep_authenticated` — discard guest cart, return auth cart unchanged
- `keep_guest` — replace auth cart items with guest cart items

### Orders

```python
await svc.create_order(data: OmcOrderCreate) -> OmcOrderResponse
await svc.get_order(tenant_id, order_id) -> OmcOrderResponse | None
await svc.update_order(tenant_id, order_id, data: OmcOrderUpdate) -> OmcOrderResponse | None
await svc.cancel_order(tenant_id, order_id, reason, by) -> OmcOrderResponse | None
await svc.list_orders(tenant_id, channel_id=None, status=None) -> list[OmcOrderResponse]
await svc.mark_order_shipped(tenant_id, order_id, tracking, by) -> OmcOrderResponse | None
await svc.mark_order_collected(tenant_id, order_id, by) -> OmcOrderResponse | None
await svc.mark_collection_ready(tenant_id, order_id, by) -> OmcOrderResponse | None
await svc.fraud_screen_order(order_id) -> dict
await svc.calculate_shipping(order_id, carrier_options=None) -> list[dict]
```

**fraud_screen_order** assembles a risk feature vector (order value, customer velocity, channel, payment method, new-customer flag) and scores it 0–1. Orders with `score >= RETAIL_OMC_FRAUD_THRESHOLD` are auto-held with status `fraud_review`.

**calculate_shipping** returns ranked carrier rate quotes. For click-and-collect orders it returns a zero-rate in-store-pickup option automatically.

### Fulfilment Workflows

```python
await svc.bopis_order(customer_id, sku, quantity, pickup_store, pickup_date) -> dict
await svc.ship_from_store(order_id, fulfilling_store_id) -> dict
await svc.order_routing(order_id, routing_rules) -> dict
await svc.click_and_collect_ready(order_id) -> dict
```

**order_routing rules**:
```python
{
    "prefer_store": True,
    "max_distance_km": 50,
    "split_allowed": True,
    "priority": "speed"  # or "cost" or "availability"
}
```

### Returns and RMA

```python
await svc.initiate_return(data: OmcReturnCreate) -> OmcReturnResponse
await svc.get_return(tenant_id, return_id) -> OmcReturnResponse | None
await svc.approve_return(tenant_id, return_id, refund_amount, by) -> OmcReturnResponse | None
await svc.list_returns(tenant_id, order_id=None) -> list[OmcReturnResponse]
await svc.omnichannel_returns(order_id, return_channel) -> dict
await svc.process_rma(return_id, received_items, condition_grades) -> dict
```

**process_rma** grades each returned item and routes it accordingly:

| Grade | Disposition | Recovery Rate |
|---|---|---|
| `new` | restock | 100% |
| `refurbished` | refurbishment queue | 70% |
| `damaged` | clearance | 30% |
| `scrap` | write-off | 0% |

```python
result = await svc.process_rma(
    return_id="rtn-001",
    received_items=[{"sku": "SKU-001", "quantity": 1, "unit_price": 4500.0}],
    condition_grades={"SKU-001": "refurbished"},
)
# result["total_refund"] == 3150.0  (70% of 4500)
```

### Customer Journey and Attribution

```python
await svc.customer_journey_event(customer_id, event_type, channel, context) -> dict
await svc.record_journey_event(data: OmcJourneyEventCreate) -> OmcJourneyEventResponse
await svc.get_session_journey(tenant_id, session_id) -> list[OmcJourneyEventResponse]
await svc.channel_attribution(order_id) -> dict          # last-touch only
await svc.multi_touch_attribution(order_id, model, time_decay_half_life_hours) -> dict
await svc.unified_customer_profile(customer_id) -> dict
```

**multi_touch_attribution models**:
- `last_touch` — 100% credit to the final pre-order touchpoint
- `first_touch` — 100% credit to the first touchpoint
- `linear` — equal credit split across all touchpoints
- `time_decay` — exponential decay; recent touches receive more credit. `time_decay_half_life_hours` controls the decay rate (default 24h)

### Loyalty Hooks

```python
await svc.earn_loyalty_points(order_id, program_id, points_per_currency_unit=1.0) -> dict
await svc.burn_loyalty_points(cart_id, points, program_id, point_value=0.01) -> OmcCartResponse | None
```

Points can only be earned on fulfilled orders (status `collected`, `delivered`, or `shipped`).

### Pricing Rules

```python
await svc.create_pricing_rule(data: OmcPricingRuleCreate) -> OmcPricingRuleResponse
await svc.list_pricing_rules(tenant_id, channel_id=None) -> list[OmcPricingRuleResponse]
await svc.apply_pricing_rules(tenant_id, sku, base_price, channel_id) -> float
```

### Analytics

```python
await svc.omnichannel_analytics(period) -> dict   # period format: "2026-06"
```

Returns: order count, total revenue, orders by channel, revenue by channel, fulfilment mode mix, return rate, journey stats, BOPIS metrics, attribution count.

### Audit Log

```python
await svc.query_audit_log(entity_type, entity_id=None, limit=100) -> list[dict]
await svc._emit_audit_event(entity_type, entity_id, action, before, after, actor=None)
```

Audit events are structured as CloudEvents for Bytewax stream compatibility.

---

## UI Routes

| Path | Permission | Nav Group |
|---|---|---|
| `/retail-omc/dashboard` | `retail_omc:view` | Overview |
| `/retail-omc/orders` | `retail_omc:view` | Orders |
| `/retail-omc/orders/<id>` | `retail_omc:view` | Orders |
| `/retail-omc/orders/create` | `retail_omc:write` | Orders |
| `/retail-omc/inventory` | `retail_omc:view` | Inventory |
| `/retail-omc/channels` | `retail_omc:admin` | Channels |
| `/retail-omc/fulfilment` | `retail_omc:write` | Fulfilment |
| `/retail-omc/carts` | `retail_omc:view` | Commerce |

---

## Business Rules

| Rule | Condition | Effect |
|---|---|---|
| `oversell_denied` | `available_stock=0` on reserve | deny |
| `fraud_hold` | `fraud_score >= FRAUD_THRESHOLD` | auto-hold |
| `payment_requires_fraud_check` | `fraud_check_passed=False` on payment | deny |
| `click_and_collect_requires_store` | C&C without `store_id` | deny |
| `channel_price_arbitrage_denied` | price arbitrage detected | deny |
| `pci_compliance_required` | non-PCI payment processor | deny |
| `return_window_expired` | return after window | deny |
| `cart_max_items_exceeded` | over item limit | deny |
| `loyalty_earn_requires_fulfilment` | earn on un-fulfilled order | deny |
| `safety_stock_reorder_trigger` | `available <= reorder_point` | emit alert |

---

## Composability

```apg
use retail_omc;
```

| Peer Capability | Integration |
|---|---|
| `retail_pos` | Calls `reserve_inventory` on POS transaction |
| `retail_loy` | Receives `earn_loyalty_points` / `burn_loyalty_points` events |
| `retail_prm` | Provides promotion discounts applied to cart line items |
| `retail_sin` | Consumes journey events for conversion funnel analysis |
| `wflo` | Receives high-fraud orders for manual review workflow |
| `moni` | Receives channel health degradation events |
| `nlpc` | Augments `search_catalogue` with NLP query expansion |

---

## Further Reading

- `service.py` — Complete async business logic
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoint definitions
- `views.py` — Flask-AppBuilder views
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 architectural improvement proposals
- `SPECIFICATION.md` — Detailed capability specification
- `cap_spec.md` — Capability contract specification
