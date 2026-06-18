# ins_act Capability Specification

**Capability ID**: `ins_act`

## Description

ins_act capability for the APG platform.

## Provides

- `reserve_calculation`
- `loss_ratio_analysis`
- `pricing_models`
- `stress_testing`
- `actuarial_reporting`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
