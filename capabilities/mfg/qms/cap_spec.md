# Quality Management System Capability Specification

**Capability ID**: `mfg_qms`

## Description

Quality Management System capability for the APG platform.

## Provides

- `inspection_plan`
- `ncr_workflow`
- `capa_workflow`
- `spc_monitoring`
- `quality_reporting`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
