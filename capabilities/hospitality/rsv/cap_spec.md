# hos_rsv Capability Specification

**Capability ID**: `hos_rsv`

## Description

hos_rsv capability for the APG platform.

## Provides

- `booking_engine`
- `channel_management`
- `group_bookings`
- `availability_sync`
- `cancellation_management`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
