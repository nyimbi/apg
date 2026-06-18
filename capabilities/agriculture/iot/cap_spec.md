# agr_iot Capability Specification

**Capability ID**: `agr_iot`

## Description

agr_iot capability for the APG platform.

## Provides

- `sensor_data_ingestion`
- `drone_imagery`
- `yield_mapping`
- `soil_analysis`
- `precision_recommendations`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
