# Enterprise Asset Management Specification

## Intent

Enterprise Asset Management (`eam_ast`) makes physical asset operations a composable APG capability. It provides executable lifecycle surfaces for locations, asset records, criticality, condition readings, maintenance plans, work orders, inspections, inventory reservations, asset reliability analytics, AI-agent review, UI routes, theming, deterministic rules, and Bytewax lifecycle streaming.

The capability is designed for generated APG applications that need to track equipment, sites, facilities, fleet assets, production machinery, tooling, and capital assets while remaining dependency-light enough to compile and execute in package tests.

## Functional Requirements

- Register tenant-scoped locations with optional parent hierarchy.
- Register tenant-scoped assets with owner, category, location, criticality, health score, capitalisation flag, and fixed-asset reference when required.
- Create maintenance plans with strategy, positive interval, and condition source for predictive strategies.
- Open work orders against assets with priority, safety plan, and approval for critical assets.
- Complete work orders only when an outcome is recorded.
- Record inspections with result, inspector, optional condition score, and asset health update.
- Record condition readings with metric, value, unit, optional alert threshold, and review requirement for alerting readings.
- Reserve inventory parts with positive quantity and optional work-order linkage.
- Register first-class EAM agents for Codex, Claude Code, OpenCode, and Pi.
- Validate privileged AI-agent maintenance actions through a human approval guardrail.
- Expose dashboard, location, asset, maintenance-plan, work-order, inspection, inventory, analytics, agent, and settings UI route metadata.
- Emit lifecycle events through a Bytewax-backed stream named `apg.eam.ast.lifecycle`.

## Rule Engine

The deterministic rule engine evaluates plain context dictionaries and returns `allow`, `deny`, or `require_review`. It enforces tenant context, write policy attachment, location type, asset owner/category/location/criticality, fixed-asset references for capital assets, health-score bounds, maintenance strategy and interval, predictive condition source, work-order asset/priority/safety/approval, completion outcome, inspection result, condition metric/value/review, inventory part/quantity, Bytewax routing, supported EAM-agent runtime and role, and human approval for privileged agent actions.

## Configuration

The contract exposes explicit configuration sections:

- `assets`
- `locations`
- `maintenance_plans`
- `work_orders`
- `inspections`
- `inventory`
- `analytics`
- `eam_agents`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

Tenant overrides are passed to `get_capability_contract(tenant_id, overrides)` and deep-merged into the default configuration.

## Composition Interfaces

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

## Acceptance Criteria

- `get_capability_contract()` returns a valid APG contract with configuration, schema, deterministic rules, UI routes, theme tokens, and Bytewax streaming metadata.
- Package import exposes `EnterpriseAssetManagementService`, `EAMAssetService`, contract helpers, streaming metadata, and registration metadata without requiring optional web or database dependencies.
- Service supports location, asset, maintenance-plan, work-order, inspection, condition-reading, inventory-reservation, EAM-agent, dashboard, analytics, audit, import-validation, and compatibility record operations.
- API helpers and view models expose the same lifecycle surfaces.
- Semantic model includes EAM-agent metadata, required dependencies, route metadata, rules, theme, and Bytewax stream metadata.
- Focused tests cover lifecycle success paths, guardrail failures, API/view execution, app self-test, and semantic metadata.

