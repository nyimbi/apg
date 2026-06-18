# Advanced Planning and Scheduling Capability Specification

**Capability ID**: `mfg_aps`

## Description

Advanced Planning and Scheduling capability for the APG platform.

## Provides

- `finite_capacity_scheduling`
- `gantt_visualisation`
- `sequence_optimisation`
- `constraint_dispatch`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
