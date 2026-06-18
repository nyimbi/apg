# agr_ins Capability Specification

**Capability ID**: `agr_ins`

## Description

agr_ins capability for the APG platform.

## Provides

- `policy_issuance`
- `parametric_claims`
- `satellite_verification`
- `mobile_payout`
- `index_calculation`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
