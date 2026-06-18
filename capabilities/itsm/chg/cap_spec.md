# itsm_chg Capability Specification

**Capability ID**: `itsm_chg`

## Description

itsm_chg capability for the APG platform.

## Provides

- `change_lifecycle_workflow`
- `cab_approval_workflow`
- `change_schedule_management`
- `change_conflict_detection`
- `post_implementation_review`
- `emergency_change_workflow`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
