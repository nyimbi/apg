# hos_spa Capability Specification

**Capability ID**: `hos_spa`

## Description

hos_spa capability for the APG platform.

## Provides

- `spa_booking`
- `treatment_management`
- `activity_scheduling`
- `gift_vouchers`
- `therapist_management`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
