# hos_fdb Capability Specification

**Capability ID**: `hos_fdb`

## Description

hos_fdb capability for the APG platform.

## Provides

- `restaurant_pos`
- `menu_management`
- `table_reservations`
- `cost_control`
- `fb_inventory`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
