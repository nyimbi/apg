# hos_pms Capability Specification

**Capability ID**: `hos_pms`

## Description

hos_pms capability for the APG platform.

## Provides

- `room_management`
- `check_in_out`
- `housekeeping_management`
- `folio_management`
- `room_assignment`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
