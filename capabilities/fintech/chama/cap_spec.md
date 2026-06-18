# Chama & ROSCA Engine Capability Specification

**Capability ID**: `fintech_chama`

## Description

Chama & ROSCA Engine capability for the APG platform.

## Provides

- `chama_management`
- `rosca_rotation`
- `group_lending`
- `treasury_management`
- `mobile_disbursement`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
