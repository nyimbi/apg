# ins_prm Capability Specification

**Capability ID**: `ins_prm`

## Description

ins_prm capability for the APG platform.

## Provides

- `premium_collection`
- `instalment_scheduling`
- `payment_reminders`
- `lapse_management`
- `refund_processing`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
