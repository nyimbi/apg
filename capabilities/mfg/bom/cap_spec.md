# Bill of Materials Capability Specification

**Capability ID**: `mfg_bom`

## Description

Bill of Materials capability for the APG platform.

## Provides

- `bom_structure`
- `bom_explosion`
- `eco_workflow`
- `bom_comparison`
- `cost_rollup`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
