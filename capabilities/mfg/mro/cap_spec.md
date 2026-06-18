# Maintenance, Repair and Overhaul Capability Specification

**Capability ID**: `mfg_mro`

## Description

Maintenance, Repair and Overhaul capability for the APG platform.

## Provides

- `maintenance_work_order`
- `pm_scheduling`
- `failure_analysis`
- `spare_parts_management`
- `asset_uptime_tracking`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
