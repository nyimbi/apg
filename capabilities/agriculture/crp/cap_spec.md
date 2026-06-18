# agr_crp Capability Specification

**Capability ID**: `agr_crp`

## Description

agr_crp capability for the APG platform.

## Provides

- `crop_registration`
- `planting_calendar`
- `phenology_tracking`
- `yield_recording`
- `variety_registry`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
