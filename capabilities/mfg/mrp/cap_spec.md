# Material Requirements Planning Capability Specification

**Capability ID**: `mfg_mrp`

## Description

Material Requirements Planning capability for the APG platform.

## Provides

- `mrp_planning_run`
- `production_order_workflow`
- `purchase_requisition_workflow`
- `demand_pegging`
- `exception_message_workflow`
- `net_change_planning`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
