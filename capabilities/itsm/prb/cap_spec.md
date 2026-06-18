# itsm_prb Capability Specification

**Capability ID**: `itsm_prb`

## Description

itsm_prb capability for the APG platform.

## Provides

- `problem_lifecycle_workflow`
- `rca_workflow`
- `known_error_database`
- `workaround_management`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
