# Enterprise Asset Management Capability Summary

`eam_ast` is a dependency-light APG package for composing executable asset-management applications. It covers location hierarchy, asset registry, maintenance planning, work orders, inspections, condition readings, inventory reservations, asset reliability analytics, AI-agent review, deterministic guardrails, UI route metadata, theme tokens, and Bytewax lifecycle streaming.

## Lifecycle

1. Register a tenant location.
2. Register an asset with owner, category, location, criticality, and optional capital-asset reference.
3. Create a maintenance plan with strategy, interval, and condition source when predictive.
4. Open a work order with safety plan and critical-asset approval when required.
5. Reserve parts for the work order.
6. Record inspection and condition readings.
7. Complete the work order with an outcome.
8. Review reliability analytics and audit events.
9. Register EAM agents that can recommend, validate, and prepare maintenance actions under human-approval guardrails.

## Composition Surface

Provides:

- `asset_registry_lifecycle`
- `asset_location_hierarchy`
- `criticality_and_condition_management`
- `maintenance_plan_lifecycle`
- `work_order_lifecycle`
- `inspection_and_condition_readings`
- `asset_reliability_analytics`
- `eam_agents`

Requires:

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `fixed_asset_management`

## Runtime Entry Points

- `capability_contract.py`: APG contract, rules, UI, theme, and Bytewax stream metadata.
- `service.py`: executable EAM domain service.
- `api.py`: dependency-light API helper functions.
- `views.py`: UI view models for APG composition.
- `app.py`: semantic model, component manifest, and self-test.
- `tests/test_package_contract.py`: focused package verification.
