# Budgeting and Forecasting

`bfc_budgeting_forecasting` is the APG capability for composing budget planning, financial forecasting, scenario planning, variance analysis, planning collaboration, and budget approval workflows into generated Python applications. It provides an executable service surface, deterministic guardrails, UI metadata, theme metadata, and Bytewax lifecycle-stream declarations.

## What It Provides

- Budget creation with owner, fiscal year, currency, and period controls.
- Budget line management for revenue, expense, capital, and headcount plans.
- Budget submission and approval with separation-of-duties controls.
- Forecast creation with method, horizon, confidence, and forecast points.
- Scenario planning with probability and driver assumptions.
- Variance analysis with threshold-based review controls.
- Collaboration sessions for planning work.
- First-class BFC agents for Codex, Claude Code, OpenCode, and Pi.
- Deterministic rules for tenant, policy, planning, financial, agent, and stream guardrails.
- Bytewax lifecycle stream metadata.
- UI route and theme metadata for APG composition.

## Quick Start

```python
from capabilities.fin.bfc.budgeting_forecasting import BudgetingForecastingService

service = BudgetingForecastingService()
budget = service.create_budget(
    "budget-2026",
    "tenant-a",
    "FY2026 Operating Budget",
    "finance",
    2026,
    "USD",
    "2026-01-01",
    "2026-12-31",
)
service.add_budget_line(
    "line-1",
    "tenant-a",
    budget["id"],
    "4000",
    "revenue",
    250000,
    "2026",
)
submitted = service.submit_budget("tenant-a", budget["id"], "finance")
approved = service.approve_budget("tenant-a", submitted["id"], "controller")
forecast = service.create_forecast(
    "forecast-1",
    "tenant-a",
    "Q1 Rolling Forecast",
    "trend",
    12,
    confidence=82,
    base_budget_record_id=approved["id"],
)
service.record_forecast_point("point-1", "tenant-a", forecast["id"], "2026-01", 21000)
summary = service.dashboard_summary("tenant-a")
```

## Contract

Use `get_capability_contract()` to inspect the APG composition surface.

```python
from capabilities.fin.bfc.budgeting_forecasting import get_capability_contract

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
- Budgets without owner, fiscal year, currency, or valid period dates.
- Budget lines without budget, account, supported line type, or positive amount.
- Budget submission without at least one line.
- Budget approval without submitted state, approval evidence, or independent approver.
- High-value budgets without review.
- Forecasts with unsupported method, invalid horizon, or confidence outside 0 to 100.
- Forecast points without forecast or period.
- Scenarios without name, driver assumptions, or valid probability.
- Variances without budget, actual amount, or review for material variance.
- Collaboration sessions without budget or participants.
- Batch and lifecycle events not routed through Bytewax.
- Unsupported BFC-agent runtime or role.
- Privileged BFC-agent actions without human approval.

## UI And Theme

The capability publishes route metadata for:

- `/bfc-budgeting-forecasting/dashboard`
- `/bfc-budgeting-forecasting/budgets`
- `/bfc-budgeting-forecasting/budget-lines`
- `/bfc-budgeting-forecasting/forecasts`
- `/bfc-budgeting-forecasting/scenarios`
- `/bfc-budgeting-forecasting/variances`
- `/bfc-budgeting-forecasting/approvals`
- `/bfc-budgeting-forecasting/collaboration`
- `/bfc-budgeting-forecasting/agents`
- `/bfc-budgeting-forecasting/settings`

The default theme is `bfc_budgeting_forecasting_control`. View helpers in `views.py` return dashboard, budget, budget-line, forecast, scenario, variance, approval, collaboration, and agent workbench models.

## AI Agents

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `budget_planning_reviewer`
- `forecast_reviewer`
- `variance_reviewer`
- `scenario_reviewer`
- `approval_reviewer`
- `cash_flow_reviewer`

Register an agent with `register_bfc_agent()` and validate privileged proposals with `validate_agent_bfc_action()`.

## Verification

Focused verification for this package:

```bash
./.venv/bin/python -m py_compile \
  capabilities/fin/bfc/budgeting_forecasting/__init__.py \
  capabilities/fin/bfc/budgeting_forecasting/capability_contract.py \
  capabilities/fin/bfc/budgeting_forecasting/service.py \
  capabilities/fin/bfc/budgeting_forecasting/api.py \
  capabilities/fin/bfc/budgeting_forecasting/views.py \
  capabilities/fin/bfc/budgeting_forecasting/app.py \
  capabilities/fin/bfc/budgeting_forecasting/tests/test_package_contract.py

./.venv/bin/pytest -q capabilities/fin/bfc/budgeting_forecasting/tests/test_package_contract.py
./.venv/bin/python capabilities/fin/bfc/budgeting_forecasting/app.py
```

Deferred live-system work includes durable stores, live financial adapters, business-intelligence providers, durable Bytewax deployment, rendered browser UI, and performance testing.
