# common_mlr Capability Specification

**Capability ID**: `common_mlr`

## Description

common_mlr capability for the APG platform.

## Provides

- `experiment_tracking`
- `run_comparison`
- `artifact_versioning`
- `feature_store`
- `feature_serving`
- `point_in_time_features`
- `model_registry`
- `model_promotion`
- `ab_testing`
- `shadow_deployment`
- `drift_detection`
- `data_quality_monitoring`
- `retraining_triggers`

## Composability

This capability can be composed with `auth`, `audl`, and `notif` capabilities.

## Interfaces

- REST API via `api.py` Blueprint
- Pydantic models via `views.py`
- Service layer via `service.py`
