# Omnichannel Commerce

**Capability ID**: `retail_omc` | **Domain**: `retail` | **Version**: `1.0.0`

## Description

Provides unified commerce orchestration across all retail touchpoints: channel registry, cross-channel inventory visibility with reservation TTL, unified cart and order management, buy-online-pickup-in-store (BOPIS/C&C), ship-from-store fulfilment, multi-channel returns, customer journey event tracking with attribution, cross-channel pricing rules, and fraud screening integration. All operations are tenant-isolated and streamed to Bytewax.

## Installation

```bash
pip install apg-retail-omc
```

## Provides

- `omnichannel_order_management`
- `inventory_visibility`
- `click_and_collect`
- `customer_journey_orchestration`
- `unified_cart`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/retail-omc/dashboard` | `retail_omc:view` | Overview |
| `/retail-omc/orders` | `retail_omc:view` | Orders |
| `/retail-omc/orders/<id>` | `retail_omc:view` | Orders |
| `/retail-omc/orders/create` | `retail_omc:write` | Orders |
| `/retail-omc/inventory` | `retail_omc:view` | Inventory |
| `/retail-omc/channels` | `retail_omc:admin` | Channels |
| `/retail-omc/fulfilment` | `retail_omc:write` | Fulfilment |
| `/retail-omc/carts` | `retail_omc:view` | Commerce |

## Key Service Methods

- `create_channel()`
- `get_channel()`
- `list_channels()`
- `create_catalogue_item()`
- `get_catalogue_item()`
- `get_catalogue_item_by_sku()`
- `list_catalogue_items()`
- `set_channel_price()`
- `unified_inventory_check()`
- `upsert_inventory()`

_(See `service.py` for complete API.)_

## Interoperability

`retail_omc` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use retail_omc;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `RETAIL_OMC_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
