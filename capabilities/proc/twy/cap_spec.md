# Three-Way Match Engine Capability Specification

**Capability ID**: `proc_twy`

## Description

Three-Way Match Engine capability for the APG platform.

## Provides

- `three_way_match`
- `exception_management`
- `tolerance_rules`
- `match_analytics`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
