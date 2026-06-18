# leg_cpl Capability Specification

**Capability ID**: `leg_cpl`

## Description

leg_cpl capability for the APG platform.

## Provides

- `compliance_calendar`
- `obligations_register`
- `compliance_testing`
- `compliance_reporting`
- `breach_management`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
