# ECM / Records Management Capability Specification

**Capability ID**: `ckm_ecm`

## Description

ECM / Records Management capability for the APG platform.

## Provides

- `document_management`
- `version_control`
- `retention_management`
- `content_workflow`
- `disposal_management`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
