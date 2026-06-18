# proc_sup_portal Capability Specification

**Capability ID**: `proc_sup_portal`

## Description

proc_sup_portal capability for the APG platform.

## Provides

- `supplier_registration`
- `po_acknowledgement`
- `quote_submission`
- `invoice_submission`
- `delivery_confirmation`
- `dispute_management`
- `supplier_performance_dashboard`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
