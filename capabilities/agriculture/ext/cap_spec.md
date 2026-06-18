# agr_ext Capability Specification

**Capability ID**: `agr_ext`

## Description

agr_ext capability for the APG platform.

## Provides

- `advisory_delivery`
- `training_management`
- `farmer_outreach`
- `knowledge_base`
- `field_visit_tracking`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
