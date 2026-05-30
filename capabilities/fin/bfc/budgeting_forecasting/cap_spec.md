# Budgeting and Forecasting Capability Summary

`bfc_budgeting_forecasting` provides the APG budgeting and forecasting packet for budget planning, budget lines, approvals, forecasts, scenarios, variance analysis, collaboration, and BFC-agent composition.

## Provides

- `budget_planning_lifecycle`
- `budget_line_management`
- `budget_approval_workflow`
- `forecast_lifecycle`
- `scenario_planning`
- `variance_analysis_lifecycle`
- `planning_collaboration`
- `bfc_agents`

## Requires

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `general_ledger`
- `accounts_payable`
- `accounts_receivable`
- `cash_management`
- `business_intelligence`

## Execution Model

The package is executable without optional web or database dependencies. `BudgetingForecastingService` owns the in-memory lifecycle state, evaluates deterministic rules before writes, emits audit events with Bytewax stream metadata, and exposes summaries for generated applications.

## Composition Metadata

- Event processor: `bytewax`
- Stream: `apg.fin.bfc.lifecycle`
- Theme: `bfc_budgeting_forecasting_control`
- UI shell: `apg_python`
- App target: `python`

## Deferred Integration

Durable storage, live GL/AP/AR/cash integration, business-intelligence provider connections, authorization, notification, audit sinks, and durable Bytewax topologies remain adapter work after the executable package baseline.
