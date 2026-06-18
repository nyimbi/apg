# leg_ent Capability Specification

**Capability ID**: `leg_ent`

## Description

leg_ent capability for the APG platform.

## Provides

- `entity_registry`
- `shareholder_register`
- `board_resolutions`
- `statutory_filings`
- `cap_table`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
