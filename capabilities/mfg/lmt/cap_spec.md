# Lot and Batch Management Capability Specification

**Capability ID**: `mfg_lmt`

## Description

Lot and Batch Management capability for the APG platform.

## Provides

- `lot_creation`
- `lot_traceability`
- `shelf_life_management`
- `lot_recall`
- `genealogy_query`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
