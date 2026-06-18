# agr_irg Capability Specification

**Capability ID**: `agr_irg`

## Description

agr_irg capability for the APG platform.

## Provides

- `irrigation_scheduling`
- `water_usage_tracking`
- `sensor_integration`
- `schedule_optimisation`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
