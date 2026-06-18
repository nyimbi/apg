# leg_dsc Capability Specification

**Capability ID**: `leg_dsc`

## Description

leg_dsc capability for the APG platform.

## Provides

- `document_management`
- `legal_hold`
- `ediscovery_production`
- `privilege_log`
- `document_search`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
