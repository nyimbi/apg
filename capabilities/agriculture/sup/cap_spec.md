# agr_sup Capability Specification

**Capability ID**: `agr_sup`

## Description

agr_sup capability for the APG platform.

## Provides

- `produce_traceability`
- `cold_chain_management`
- `aggregation_management`
- `buyer_linkage`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
