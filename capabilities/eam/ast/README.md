# Enterprise Asset Management

## Overview

Enterprise Asset Management (EAM) is the APG capability for the full lifecycle of physical capital assets: facilities, fleet, rotating equipment, tooling, production plant, and infrastructure. It provides a multi-tenant, policy-governed Python service surface that covers location hierarchy, asset master data, maintenance plans, work orders, inspections, condition readings, inventory reservations, and reliability analytics — all underpinned by deterministic guardrails and a Bytewax event stream.

The business value is centralising every touchpoint of asset ownership — from acquisition and commissioning through predictive maintenance and condition monitoring to end-of-life retirement — in a single composable unit that integrates with the APG Fixed Asset Management, Predictive Maintenance, Digital Twin, Notification Engine, and AI Orchestration capabilities. The result is reduced unplanned downtime, full audit traceability, and a governed surface for AI agents to propose and review asset actions without bypassing human oversight.

## Capability ID

`eam_ast`  Version: 1.0

## Provides

| Service | Description |
|---------|-------------|
| asset_registry_lifecycle | Register, update, classify, and retire physical assets with owner, category, location, criticality, health score, and optional fixed-asset linkage |
| asset_location_hierarchy | Create and manage unlimited-depth location trees (site → building → floor → room → zone) with GPS coordinates and environmental metadata |
| criticality_and_condition_management | Classify asset criticality (low/medium/high/critical) and record condition readings with alerting thresholds and review enforcement |
| maintenance_plan_lifecycle | Create and manage preventive and predictive maintenance plans with strategy, interval, and condition-source enforcement |
| work_order_lifecycle | Open, execute, and close work orders with priority, safety plan, critical-asset approval, and outcome recording |
| inspection_and_condition_readings | Record structured inspections and sensor-based condition readings against assets with result and metric enforcement |
| asset_reliability_analytics | Compute and surface availability, MTBF, MTTR, OEE, and health-score trends for assets and fleets |
| eam_agents | Register and govern AI agents (Codex, Claude Code, OpenCode, Pi) in asset reliability, maintenance planning, inspection, safety, inventory, and lifecycle-cost reviewer roles |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | RBAC authentication and tenant-scoped identity for all asset operations |
| audl | Durable audit trail for every state-changing asset event |
| ntfy | Notification delivery for work order alerts, maintenance reminders, and condition alarms |
| composition_events | APG event bus integration for cross-capability event routing |
| composition_config | APG configuration injection and tenant override management |
| fixed_asset_management | Financial asset reference linkage for capitalized assets (fixed_asset_id enforcement) |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tenant_id | string | "default" | Tenant namespace for all asset operations |
| assets.owner_required | bool | true | Block asset registration without an assigned owner |
| assets.category_required | bool | true | Block asset registration without a lifecycle category |
| assets.location_required | bool | true | Block asset registration without a registered location |
| assets.criticality_required | bool | true | Block asset registration without a criticality classification |
| assets.fixed_asset_reference_required_when_capitalized | bool | true | Require fixed-asset reference for capitalized assets |
| assets.health_score_bounds | [int, int] | [0, 100] | Acceptable range for health score values |
| locations.location_type_required | bool | true | Block location registration without a type |
| locations.parent_validation_required | bool | true | Validate parent location exists before creating child |
| maintenance_plans.strategy_required | bool | true | Require maintenance strategy on plan creation |
| maintenance_plans.interval_required | bool | true | Require interval on plan creation |
| maintenance_plans.condition_source_required_for_predictive | bool | true | Require condition source for predictive plans |
| work_orders.asset_required | bool | true | Block work order creation without an asset |
| work_orders.priority_required | bool | true | Block work order creation without priority |
| work_orders.safety_plan_required | bool | true | Block work order creation without a safety plan |
| work_orders.approval_required_for_critical | bool | true | Require approval before opening work orders on critical assets |
| work_orders.completion_outcome_required | bool | true | Require outcome before closing a work order |
| inspections.asset_required | bool | true | Block inspection recording without an asset |
| inspections.result_required | bool | true | Block inspection recording without a result |
| inspections.condition_alert_review_required | bool | true | Require review when condition reading triggers an alert |
| inventory.part_required | bool | true | Block inventory reservation without a part reference |
| inventory.positive_quantity_required | bool | true | Block inventory reservation with quantity <= 0 |
| analytics.asset_reliability_enabled | bool | true | Enable reliability analytics computation |
| analytics.condition_health_scoring_enabled | bool | true | Enable condition-based health scoring |
| eam_agents.enabled | bool | true | Allow EAM agent registration and action proposals |
| eam_agents.human_approval_required | bool | true | Require human approval for privileged agent actions |
| eam_agents.max_autonomous_scope | string | "recommend_validate_and_prepare" | Ceiling on agent autonomy |
| governance.require_tenant_context | bool | true | Deny all operations missing tenant context |
| governance.audit_state_changes | bool | true | Emit audit events for every state change |
| governance.policy_attached_for_writes | bool | true | Require policy attachment on write operations |
| governance.safety_review_for_critical_work | bool | true | Flag critical asset work orders for safety review |
| observability.event_stream | string | "apg.eam.ast.lifecycle" | Bytewax stream name |
| observability.stream_processor | string | "bytewax" | Stream processing engine |
| ui.enable_dashboard | bool | true | Expose the asset dashboard route |
| ui.enable_agents | bool | true | Expose the EAM agent workbench route |
| theme.default_theme | string | "eam_ast_control" | Default UI theme token set |
| theme.allow_tenant_overrides | bool | true | Allow tenants to override theme tokens |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /eam-ast/dashboard | GET | eam_ast:view | Overview |
| assets | /eam-ast/assets | GET/POST | eam_ast:manage_assets | Assets |
| locations | /eam-ast/locations | GET/POST | eam_ast:manage_locations | Assets |
| maintenance_plans | /eam-ast/maintenance-plans | GET/POST | eam_ast:manage_maintenance | Maintenance |
| work_orders | /eam-ast/work-orders | GET/POST | eam_ast:manage_work_orders | Maintenance |
| inspections | /eam-ast/inspections | GET/POST | eam_ast:inspect | Reliability |
| inventory | /eam-ast/inventory | GET/POST | eam_ast:manage_inventory | Maintenance |
| analytics | /eam-ast/analytics | GET | eam_ast:analytics | Reliability |
| agents | /eam-ast/agents | GET/POST | eam_ast:admin | Automation |
| settings | /eam-ast/settings | GET/PUT | eam_ast:admin | Administration |

REST API prefix: `/eam-ast/api/v1`

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present = false | deny — attach_tenant_context |
| eam_write_requires_policy | operation_type = write AND policy_attached = false | deny — attach_operation_policy |
| location_requires_type | register_location AND location_type_present = false | deny — set_location_type |
| asset_requires_owner | register_asset AND asset_owner_assigned = false | deny — assign_asset_owner |
| asset_requires_category | register_asset AND asset_category_present = false | deny — set_asset_category |
| asset_requires_location | register_asset AND asset_location_present = false | deny — attach_asset_location |
| asset_requires_criticality | register_asset AND criticality_present = false | deny — classify_asset_criticality |
| capital_asset_requires_fixed_asset_reference | register_asset AND capitalized = true AND fixed_asset_ref_present = false | deny — attach_fixed_asset_reference |
| asset_health_score_bounds_min | register_asset AND health_score < 0 | deny — set_health_score_between_0_and_100 |
| asset_health_score_bounds_max | register_asset AND health_score > 100 | deny — set_health_score_between_0_and_100 |
| maintenance_plan_requires_strategy | create_maintenance_plan AND maintenance_strategy_present = false | deny — set_maintenance_strategy |
| maintenance_plan_requires_interval | create_maintenance_plan AND interval_present = false | deny — set_maintenance_interval |
| maintenance_plan_interval_positive | create_maintenance_plan AND interval_days <= 0 | deny — set_positive_interval |
| predictive_plan_requires_condition_source | create_maintenance_plan AND predictive_plan = true AND condition_source_present = false | deny — attach_condition_source |
| work_order_requires_asset | open_work_order AND asset_present = false | deny — attach_asset |
| work_order_requires_priority | open_work_order AND priority_present = false | deny — set_priority |
| work_order_requires_safety_plan | open_work_order AND safety_plan_present = false | deny — attach_safety_plan |
| critical_work_order_requires_approval | open_work_order AND critical_asset = true AND approved = false | require_review — record_approval |
| work_order_completion_requires_outcome | complete_work_order AND outcome_present = false | deny — record_completion_outcome |
| inspection_requires_asset | record_inspection AND asset_present = false | deny — attach_asset |
| inspection_requires_result | record_inspection AND inspection_result_present = false | deny — record_inspection_result |
| condition_reading_requires_metric | record_condition_reading AND metric_present = false | deny — set_condition_metric |
| condition_reading_requires_value | record_condition_reading AND value_present = false | deny — set_condition_value |
| condition_alert_requires_review | record_condition_reading AND condition_alert = true AND review_recorded = false | require_review — record_condition_review |
| inventory_reservation_requires_part | reserve_inventory AND part_present = false | deny — attach_part |
| inventory_reservation_requires_quantity | reserve_inventory AND quantity_present = false | deny — set_quantity |
| inventory_quantity_positive | reserve_inventory AND quantity <= 0 | deny — set_positive_quantity |
| eam_batch_import_requires_bytewax | eam_batch_import AND event_stream != bytewax | deny — route_asset_import_to_bytewax |
| eam_event_requires_bytewax | eam_event AND event_stream != bytewax | deny — route_asset_event_to_bytewax |
| eam_agent_runtime_supported | register_eam_agent AND agent_runtime_supported = false | deny — select_supported_agent_runtime |
| eam_agent_role_supported | register_eam_agent AND agent_role_supported = false | deny — select_supported_agent_role |
| privileged_agent_eam_action_requires_human_approval | agent_eam_action AND privileged_scope = true AND human_approval_recorded = false | deny — record_human_approval |

## Data Models

| Model | Key Fields |
|-------|-----------|
| EAAsset | asset_id, tenant_id, asset_number, asset_name, asset_type, asset_category, criticality_level, location_id, status, operational_status, lifecycle_stage, health_score, condition_status, maintenance_strategy, next_maintenance_due, fixed_asset_id, is_capitalized, purchase_cost, current_book_value, mtbf_hours, mttr_hours, availability_target, digital_twin_id, iot_enabled |
| EALocation | location_id, tenant_id, location_code, location_name, location_type, parent_location_id, hierarchy_level, hierarchy_path, gps_latitude, gps_longitude, floor_area_sqm, safety_zone, hazardous_area_rating, is_active |
| EAWorkOrder | work_order_id, tenant_id, work_order_number, title, work_type, priority, asset_id, location_id, status, workflow_stage, scheduled_start, scheduled_end, actual_start, actual_end, estimated_hours, actual_cost, assigned_to, safety_category, permits_required, requires_approval, approved_by, completion_percentage, failure_mode, root_cause_analysis |
| EAMaintenanceRecord | record_id, tenant_id, asset_id, work_order_id, maintenance_number, maintenance_type, maintenance_category, started_at, completed_at, duration_hours, downtime_hours, outcome, condition_before, condition_after, health_score_before, health_score_after, total_cost, first_time_fix, triggered_by_prediction, effectiveness_score |
| EAInventory | inventory_id, tenant_id, part_number, description, item_type, category, current_stock, minimum_stock, reorder_point, economic_order_quantity, unit_cost, average_cost, primary_vendor_id, lead_time_days, compatible_assets, criticality, status, auto_reorder |
| EAContract | contract_id, tenant_id, contract_number, contract_name, contract_type, contractor_id, start_date, end_date, contract_value, response_time_hours, resolution_time_hours, availability_target, status, approval_status, auto_renewal, renewal_notice_days |
| EAPerformanceRecord | record_id, tenant_id, asset_id, measurement_date, measurement_period, availability_percentage, failure_count, mean_time_between_failures, mean_time_to_repair, oee_availability, oee_performance, oee_quality, oee_overall, health_score, trend_direction, energy_consumption, co2_emissions |
| EAAssetContract | asset_id, contract_id, tenant_id, coverage_start_date, coverage_end_date, coverage_type, priority_level |

## Streaming Events

Events emitted to the eam event stream (`apg.eam.ast.lifecycle`) via Bytewax.

| Event | Trigger |
|-------|---------|
| location_registered | A new location node is successfully created in the hierarchy |
| asset_registered | A new asset passes all registration guardrails and is persisted |
| maintenance_plan_created | A maintenance plan is created with valid strategy, interval, and optional condition source |
| work_order_opened | A work order passes asset, priority, safety plan, and approval checks |
| work_order_completed | A work order is closed with a recorded outcome |
| inspection_recorded | An inspection result is captured against an asset |
| condition_reading_recorded | A sensor or manual condition reading is stored; alerts trigger review routing |
| inventory_reservation_created | A parts reservation against a work order is confirmed with valid part and positive quantity |
| eam_agent_registered | An EAM AI agent is registered with a supported runtime and role |

Asset lifecycle states tracked in stream: `draft`, `active`, `in_service`, `maintenance_due`, `work_open`, `work_complete`, `degraded`, `retired`.

## Edge Cases Handled

- Capitalized assets without a fixed-asset reference are hard-blocked at registration; the rule fires independently of the owner/category/location checks so all four can be reported in a single pass.
- Health scores are range-checked with two separate rules (< 0 and > 100) so the precise bound violation is identifiable from the rule name in audit logs.
- Predictive maintenance plans require a condition source, but the rule only fires when `predictive_plan = true`, leaving preventive plans free of that constraint without a branching code path.
- Critical work orders (criticality_level in `['high', 'critical']`) are routed to `require_review` rather than hard-denied, allowing the approval to be captured asynchronously without blocking the work order record's creation.
- Condition readings that breach an alert threshold are also routed to `require_review` rather than denied, ensuring the reading is stored for time-series continuity while the review obligation is enforced.
- Batch asset imports and lifecycle events that are not routed through Bytewax are denied outright, preventing silent data inconsistency between the relational store and the stream.
- AI agent privileged actions (scope beyond `recommend_validate_and_prepare`) are denied unless a human approval record is present, enforcing the human-in-the-loop constraint regardless of agent runtime.
- Parent location validation is enforced before child location creation, preventing orphaned hierarchy nodes even in concurrent import scenarios.
- Inventory reservations with `quantity <= 0` are blocked by a separate rule from the `quantity_present` check, allowing zero-quantity and missing-quantity errors to be distinguished in the audit trail.

## Composability

Describes how this capability integrates with others:

- **Upstream**: `fixed_asset_management` provides the financial asset register that capitalized EAM assets must reference; `auth` provides tenant-scoped RBAC identities that own and approve assets; `composition_config` injects tenant-level configuration overrides at runtime.
- **Downstream**: `predictive_maintenance` consumes asset health scores, condition readings, and maintenance records to build failure prediction models; `digital_twin_marketplace` mirrors asset state for simulation; `notification_engine` receives work order and condition alert events to dispatch stakeholder communications; `audit_compliance` consumes the full state-change event stream for regulatory reporting.
- **Peer**: `life_cycle_costing` (`eam/lcc`) aggregates maintenance and operational costs per asset for TCO analysis; `maintenance_scheduling` (`eam/msc`) uses maintenance plan metadata to produce optimised maintenance schedules; `work_order_management` (`eam/wom`) extends the work order surface with procurement and contractor management.

## Development Notes

- The service layer (`service.py`) is an in-memory coordinator; persistence adapters for PostgreSQL, the fixed-asset bridge, and the Bytewax deployment are deferred and injected via the `adapters` configuration block.
- All model prefixes use `ea_` (e.g. `EAAsset`, `ea_asset`) to namespace tables within a shared PostgreSQL schema.
- The rule engine in `capability_contract.py` is fully deterministic — no external calls, no async — so it can be used at import time for contract validation and in tests without a running database.
- `evaluate_capability_rules(context)` returns a merged decision across all matched rules: the first `deny` wins over any `require_review`, and `require_review` wins over `allow`. This allows a single context dict to be checked once and return a complete action list.
- UUID7 string IDs are used throughout (`uuid7str`) for time-ordered sortability, which aligns with Bytewax stream key ordering.
- The `EAAsset.update_search_vector()` method produces a denormalised text column for full-text search; callers must invoke it explicitly after field mutations — it is not a SQLAlchemy event listener.
- `EALocation.get_distance_to()` uses the Haversine formula and returns kilometres; it is a convenience method for field dispatch routing, not a geospatial index substitute.
- `EAPerformanceRecord.calculate_oee()` divides the product of three percentage values by 10,000 (not 100) to account for each factor already being in percent form; this is intentional and matches the ISO 22400 OEE definition.
- Theme token `eam_ast_control` uses a steel-blue primary (`#28536B`) with a high-contrast accent red (`#C44536`) to distinguish critical-asset indicators from general UI chrome.

---

## World-Class Enhancements (v2.0)

- **I1.** 10 High-Impact Improvements for World-Class EAM Solution
- **I2.** Executive Summary
- **I3.** Autonomous Maintenance Orchestration with AI Decision Trees
- **I4.** Current State
- **I5.** Proposed Enhancement
- **I6.** Temporal Asset Intelligence with Time-Series Forecasting
- **I7.** Current State
- **I8.** Proposed Enhancement
- **I9.** Immersive Mixed Reality Maintenance Guidance
- **I10.** Current State
- **I11.** Proposed Enhancement
- **I12.** Quantum-Inspired Optimization for Resource Allocation
- **I13.** Current State
- **I14.** Proposed Enhancement
- **I15.** Swarm Intelligence for Distributed Asset Monitoring

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
