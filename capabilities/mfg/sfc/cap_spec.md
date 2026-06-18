# Shop Floor Control Capability Specification

**Capability ID**: `mfg_sfc`

## Description

Shop Floor Control capability for the APG platform.

## Provides

- `routing_management`
- `work_centre_dispatch`
- `operation_tracking`
- `labour_recording`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
