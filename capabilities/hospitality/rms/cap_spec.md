# hos_rms Capability Specification

**Capability ID**: `hos_rms`

## Description

hos_rms capability for the APG platform.

## Provides

- `dynamic_pricing`
- `demand_forecasting`
- `rate_management`
- `rate_parity`
- `yield_optimisation`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
