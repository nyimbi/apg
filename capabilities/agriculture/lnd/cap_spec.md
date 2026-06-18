# agr_lnd Capability Specification

**Capability ID**: `agr_lnd`

## Description

agr_lnd capability for the APG platform.

## Provides

- `parcel_registration`
- `tenure_registry`
- `boundary_mapping`
- `land_dispute_management`
- `title_management`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
