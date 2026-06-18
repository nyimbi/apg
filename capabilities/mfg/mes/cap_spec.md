# Manufacturing Execution System Capability Specification

**Capability ID**: `mfg_mes`

## Description

Manufacturing Execution System capability for the APG platform.

## Provides

- `work_order_execution`
- `production_event_tracking`
- `oee_calculation`
- `resource_monitoring`
- `labour_transaction`
- `material_transaction`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
