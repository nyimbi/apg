# ngo_me Capability Specification

**Capability ID**: `ngo_me`

## Description

ngo_me capability for the APG platform.

## Provides

- `indicator_tracking`
- `data_collection`
- `evaluation_management`
- `impact_reporting`
- `log_frame`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
