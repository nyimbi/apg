# Omnichannel Commerce

## Overview
Provides unified commerce orchestration across all retail touchpoints: channel registry, cross-channel inventory visibility with reservation TTL, unified cart and order management, buy-online-pickup-in-store (BOPIS/C&C), ship-from-store fulfilment, multi-channel returns, customer journey event tracking with attribution, cross-channel pricing rules, and fraud screening integration. All operations are tenant-isolated and streamed to Bytewax.

## Capability ID
`retail_omc`

## Provides
| Service | Description |
|---|---|
| omnichannel_order_management | Unified order lifecycle across all channels |
| inventory_visibility | Real-time cross-location stock with reservation |
| click_and_collect | BOPIS fulfilment with store assignment |
| customer_journey_orchestration | Stage-by-stage journey event tracking |
| unified_cart | Channel-agnostic cart with promotion application |
| cross_channel_fulfilment | Ship-from-store, C&C, and home delivery routing |
| omnichannel_search | Faceted catalogue search with BM25 relevance ranking |
| return_management | Cross-channel return initiation, RMA processing, and refund approval |
| channel_pricing_engine | Per-channel price overrides with arbitrage prevention |
| multi_touch_attribution | Linear, time-decay, first/last-touch attribution models |
| fraud_screening | Heuristic + ML fraud scoring with auto-hold on threshold breach |
| shipping_rate_engine | Multi-carrier rate calculation with weight and zone pricing |
| loyalty_composability | Earn/burn lifecycle hooks into retail_loy |
| safety_stock_management | Demand-volatility-based safety stock with low-stock alerting |
| audit_trail | Structured CloudEvent audit log with entity-level queryability |
| cart_merge | Guest-to-authenticated cart merge with union/keep strategies |

## Requires
| Capability | Reason |
|---|---|
| auth | Customer and operator authentication |
| audl | Order and inventory audit trail |
| mten | Tenant context isolation |
| conf | Channel and fulfilment configuration |
| ntfy | Order status notifications |
| wflo | High-value order approval workflow |
| mqeb | Bytewax inventory sync stream |
| moni | Inventory and fulfilment SLA monitoring |
| nlpc | Search NLP and personalisation |
| schd | Inventory sync scheduling |

## Configuration
| Key | Default | Description |
|---|---|---|
| inventory.reservation_ttl_seconds | 900 | Inventory reservation timeout |
| inventory.safety_stock_enabled | true | Safety stock buffer active |
| cart.abandonment_timeout_minutes | 60 | Cart abandonment threshold |
| cart.max_items | 200 | Maximum items per cart |
| orders.approval_required_above_value | 50,000 | High-value order approval threshold |
| returns.return_window_days | 30 | Maximum days post-purchase for returns |
| payments.pci_compliant_required | true | PCI compliance mandatory |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /retail-omc/api/v1/channels | GET/POST | List/create channels | retail_omc:view/admin |
| /retail-omc/api/v1/catalogue | GET/POST | List/create catalogue items | retail_omc:view/write |
| /retail-omc/api/v1/catalogue/search | POST | Faceted catalogue search | retail_omc:view |
| /retail-omc/api/v1/catalogue/<id>/price | PUT | Set channel price | retail_omc:admin |
| /retail-omc/api/v1/inventory | GET/POST | Get/upsert inventory | retail_omc:view/write |
| /retail-omc/api/v1/inventory/reserve | POST | Reserve inventory | retail_omc:write |
| /retail-omc/api/v1/inventory/alerts | GET | Low-stock alert list | retail_omc:view |
| /retail-omc/api/v1/inventory/safety-stock | POST | Compute safety stock for SKU/location | retail_omc:write |
| /retail-omc/api/v1/orders | GET/POST | List/create orders | retail_omc:view/write |
| /retail-omc/api/v1/orders/<id> | GET/PUT/DELETE | Order detail/update/cancel | retail_omc:view/write |
| /retail-omc/api/v1/orders/<id>/fraud-screen | POST | Run fraud screening on order | retail_omc:write |
| /retail-omc/api/v1/orders/<id>/shipping | POST | Get carrier rate quotes | retail_omc:write |
| /retail-omc/api/v1/orders/<id>/attribution | POST | Compute multi-touch attribution | retail_omc:view |
| /retail-omc/api/v1/carts/<id>/merge | POST | Merge guest cart into auth cart | retail_omc:write |
| /retail-omc/api/v1/carts/<id>/loyalty/earn | POST | Record loyalty earn for order | retail_omc:write |
| /retail-omc/api/v1/carts/<id>/loyalty/burn | POST | Apply loyalty point burn to cart | retail_omc:write |
| /retail-omc/api/v1/returns | GET/POST | List/initiate returns | retail_omc:view/write |
| /retail-omc/api/v1/returns/<id>/approve | PUT | Approve return + refund | retail_omc:write |
| /retail-omc/api/v1/returns/<id>/rma | POST | Process RMA with condition grading | retail_omc:write |
| /retail-omc/api/v1/pricing | GET/POST | List/create pricing rules | retail_omc:admin |
| /retail-omc/api/v1/pricing/apply | POST | Apply pricing to SKU | retail_omc:write |
| /retail-omc/api/v1/audit | GET | Query structured audit log | retail_omc:admin |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| oversell_denied | available_stock=0 on reserve | deny |
| payment_requires_fraud_check | fraud_check_passed=False | deny |
| click_and_collect_requires_store | C&C without store_id | deny |
| channel_price_arbitrage_denied | price arbitrage detected | deny |
| pci_compliance_required | non-PCI payment processor | deny |
| return_window_expired | return after window | deny |
| cart_max_items_exceeded | over item limit | deny |
| inventory_reservation_ttl_enforced | no TTL set | deny |

## Data Models
| Model | Key Fields |
|---|---|
| OmcChannelResponse | id, name, channel_type, currency_code |
| OmcCatalogueItemResponse | id, sku, base_price, channel_prices |
| OmcInventoryResponse | id, sku, location_id, available_qty, reserved_qty |
| OmcCartResponse | id, state, items, grand_total |
| OmcOrderResponse | id, order_number, status, fulfilment_mode, grand_total |
| OmcReturnResponse | id, return_number, status, refund_amount |
| OmcJourneyEventResponse | id, journey_stage, event_type, session_id |
| OmcPricingRuleResponse | id, rule_type, adjustment_type, adjustment_value |

## Streaming Events
- `order_created`, `order_paid`, `order_shipped`, `order_delivered`, `order_collected`, `order_cancelled`
- `inventory_reserved`, `inventory_released`
- `cart_abandoned`, `cart_converted`
- `return_initiated`, `refund_processed`
- `journey_stage_advanced`

## Edge Cases Handled
- C&C without store assignment: assertion at service layer
- Oversell: reserve returns False, no stock mutation
- Inventory reservation TTL: enforced by rule engine
- Cross-tenant order access: tenant_id check on all reads
- PCI non-compliance: rule engine blocks payment processing
- Channel price arbitrage: rule blocks misaligned pricing
- Return after window: rule requires manager exception

## Composability Notes
- **retail_pos** uses inventory reservation on transaction post
- **retail_loy** receives `earn_loyalty_points` / `burn_loyalty_points` events on order fulfilment and checkout
- **retail_prm** applies promotion discounts to cart line items via `apply_promotions`
- **retail_sin** monitors conversion events from journey tracking
- **wflo** receives high-fraud-score orders for manual review via `fraud_screen_order`
- **moni** receives channel health degradation alerts from circuit-breaker thresholds
- **nlpc** augments `search_catalogue` with NLP query expansion when wired
