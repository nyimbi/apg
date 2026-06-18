# agr_fms Capability Specification

**Capability ID**: `agr_fms`

## Description

agr_fms capability for the APG platform.

## Provides

- `farm_registration`
- `parcel_management`
- `input_recording`
- `labour_tracking`
- `farm_operations`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
