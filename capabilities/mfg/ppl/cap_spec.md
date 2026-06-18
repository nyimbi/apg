# Production Planning Capability Specification

**Capability ID**: `mfg_ppl`

## Description

Production Planning capability for the APG platform.

## Provides

- `master_production_schedule`
- `sop_process`
- `rccp`
- `demand_management`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
