# ngo_prg Capability Specification

**Capability ID**: `ngo_prg`

## Description

ngo_prg capability for the APG platform.

## Provides

- `programme_management`
- `activity_scheduling`
- `budget_tracking`
- `results_framework`
- `milestone_tracking`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
