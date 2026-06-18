# ins_clm Capability Specification

**Capability ID**: `ins_clm`

## Description

ins_clm capability for the APG platform.

## Provides

- `fnol_intake`
- `claims_adjudication`
- `claims_payment`
- `fraud_referral`
- `claims_analytics`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
