# ins_mic Capability Specification

**Capability ID**: `ins_mic`

## Description

ins_mic capability for the APG platform.

## Provides

- `micro_policy_issuance`
- `ussd_enrolment`
- `parametric_claims`
- `mobile_money_payout`
- `group_insurance`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
