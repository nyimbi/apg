# Developer Portal Capability Specification

**Capability ID**: `common_devp`

## Description

Developer Portal capability for the APG platform.

## Provides

- `api_key_management`
- `developer_onboarding`
- `usage_analytics`
- `openapi_browser`
- `webhook_management`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
