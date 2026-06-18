# leg_mat Capability Specification

**Capability ID**: `leg_mat`

## Description

leg_mat capability for the APG platform.

## Provides

- `matter_management`
- `client_intake`
- `task_management`
- `deadline_tracking`
- `matter_analytics`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
