# int_dsy Capability Specification

**Capability ID**: `int_dsy`

## Description

int_dsy capability for the APG platform.

## Provides

- `sync_configuration`
- `bidirectional_sync`
- `change_data_capture`
- `field_mapping`
- `conflict_resolution`
- `sync_monitoring`
- `sync_scheduling`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
