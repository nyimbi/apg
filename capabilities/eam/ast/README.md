# Enterprise Asset Management

`eam_ast` is the APG capability for composing asset-heavy applications: facilities, fleet, production equipment, tooling, plants, and capital assets. It provides an executable Python package surface for the asset lifecycle plus the guardrails needed to embed that lifecycle into larger generated applications.

## What It Provides

- Location hierarchy registration.
- Asset registration with owner, category, location, criticality, health score, and fixed-asset linkage.
- Maintenance-plan creation for preventive and predictive maintenance.
- Work-order opening and completion with safety and approval controls.
- Inspection and condition-reading capture.
- Inventory reservation against work orders.
- Reliability dashboard and analytics view models.
- First-class EAM agents for Codex, Claude Code, OpenCode, and Pi.
- Deterministic rules for operational guardrails.
- Bytewax lifecycle stream metadata.
- UI route and theme metadata for APG composition.

## Quick Start

```python
from capabilities.eam.ast import EnterpriseAssetManagementService

service = EnterpriseAssetManagementService()
location = service.register_location("plant-1", "tenant-a", "Plant 1", "site")
asset = service.register_asset(
    "pump-1",
    "tenant-a",
    "Main transfer pump",
    "operations",
    "rotating_equipment",
    location["location_id"],
    "critical",
    health_score=92,
    capitalized=True,
    fixed_asset_ref="fa-001",
)
plan = service.create_maintenance_plan(
    "plan-1",
    "tenant-a",
    asset["id"],
    "predictive",
    30,
    condition_source="vibration_sensor",
)
work_order = service.open_work_order(
    "wo-1",
    "tenant-a",
    asset["id"],
    "Inspect pump vibration",
    "high",
    "lockout_tagout",
    approved_by="safety-1",
)
service.reserve_inventory("res-1", "tenant-a", "seal-kit", 2, work_order["id"])
service.record_condition_reading(
    "reading-1",
    "tenant-a",
    asset["id"],
    "vibration",
    4.2,
    "mm/s",
    review_recorded=True,
    alert_threshold=3.5,
)
summary = service.dashboard_summary("tenant-a")
```

## Contract

Use `get_capability_contract()` to inspect the APG composition surface.

```python
from capabilities.eam.ast import get_capability_contract

contract = get_capability_contract("tenant-a")
print(contract["provides"])
print(contract["streaming"]["processor"])
```

The contract exposes:

- `configuration`
- `configuration_schema`
- `rule_engine`
- `ui`
- `theme`
- `streaming`

## Guardrails

The rule engine blocks or routes review for:

- Missing tenant context.
- Writes without policy attachment.
- Assets without owner, category, location, or criticality.
- Capital assets without fixed-asset reference.
- Health scores outside the accepted range.
- Maintenance plans without strategy, interval, or predictive condition source.
- Work orders without asset, priority, safety plan, or critical-asset approval.
- Work-order completion without outcome.
- Inspections without asset or result.
- Condition readings without metric or value.
- Alerting condition readings without review.
- Inventory reservations without part or positive quantity.
- Batch imports and lifecycle events not routed through Bytewax.
- Unsupported EAM-agent runtime or role.
- Privileged agent actions without human approval.

## UI And Theme

The capability publishes route metadata for:

- `/eam-ast/dashboard`
- `/eam-ast/assets`
- `/eam-ast/locations`
- `/eam-ast/maintenance-plans`
- `/eam-ast/work-orders`
- `/eam-ast/inspections`
- `/eam-ast/inventory`
- `/eam-ast/analytics`
- `/eam-ast/agents`
- `/eam-ast/settings`

The default theme is `eam_ast_control`. View helpers in `views.py` return dashboard, asset, location, maintenance-plan, work-order, inspection, inventory, analytics, condition-reading, and agent workbench models.

## AI Agents

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `asset_reliability_reviewer`
- `maintenance_planner`
- `inspection_reviewer`
- `safety_reviewer`
- `inventory_reviewer`
- `lifecycle_cost_reviewer`

Register an agent with `register_eam_agent()` and validate privileged proposals with `validate_agent_eam_action()`.

## Verification

Focused verification for this package:

```bash
./.venv/bin/python -m py_compile \
  capabilities/eam/ast/__init__.py \
  capabilities/eam/ast/capability_contract.py \
  capabilities/eam/ast/service.py \
  capabilities/eam/ast/api.py \
  capabilities/eam/ast/views.py \
  capabilities/eam/ast/app.py \
  capabilities/eam/ast/tests/test_package_contract.py

./.venv/bin/pytest -q capabilities/eam/ast/tests/test_package_contract.py
./.venv/bin/python capabilities/eam/ast/app.py
```

Deferred live-system work includes durable stores, live fixed-asset and procurement adapters, durable Bytewax deployment, browser rendering, and performance testing.

