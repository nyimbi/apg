# agr_mkt Capability Specification

**Capability ID**: `agr_mkt`

## Description

agr_mkt capability for the APG platform.

## Provides

- `produce_listing`
- `price_discovery`
- `buyer_matching`
- `escrow_payments`
- `market_analytics`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
