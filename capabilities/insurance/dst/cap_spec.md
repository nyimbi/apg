# ins_dst Capability Specification

**Capability ID**: `ins_dst`

## Description

ins_dst capability for the APG platform.

## Provides

- `agent_management`
- `commission_tracking`
- `performance_reporting`
- `licensing_registry`
- `channel_management`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
