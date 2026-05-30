# Budgeting and Forecasting Specification

## Intent

Budgeting and Forecasting (`bfc_budgeting_forecasting`) makes financial planning a composable APG capability. It provides executable lifecycle surfaces for budgets, budget lines, approvals, forecasts, forecast points, scenarios, variances, planning collaboration, BFC-agent review, UI routes, theming, deterministic rules, and Bytewax lifecycle streaming.

The capability is designed for generated APG applications that need planning operations to be executable immediately while still exposing the contract, guardrails, and metadata required for later durable storage and adapter integration.

## Functional Requirements

- Create tenant-scoped budgets with owner, fiscal year, currency, and valid planning period.
- Add tenant-scoped budget lines with budget, account, supported line type, positive amount, period, and optional cost center.
- Submit budgets only after at least one budget line exists.
- Approve submitted budgets only with approval evidence and separation of duties.
- Route high-value budget approvals for additional review.
- Create forecasts with supported method, positive horizon, configured horizon limit, confidence bounds, and optional base budget.
- Record forecast points with forecast, period, and value.
- Create scenarios with name, probability between 0 and 100, and driver assumptions.
- Record variance analysis with budget, actual amount, calculated variance amount, calculated variance percent, and review for material variances.
- Start planning collaboration sessions with budget and participants.
- Register first-class BFC agents for Codex, Claude Code, OpenCode, and Pi.
- Validate privileged AI-agent BFC actions through a human approval guardrail.
- Expose dashboard, budget, budget-line, forecast, scenario, variance, approval, collaboration, agent, and settings UI route metadata.
- Emit lifecycle events through a Bytewax-backed stream named `apg.fin.bfc.lifecycle`.

## Rule Engine

The deterministic rule engine evaluates plain context dictionaries and returns `allow`, `deny`, or `require_review`. It enforces tenant context, write policy attachment, budget owner/fiscal-year/currency/period evidence, budget-line budget/account/type/amount controls, submission line count, approval state and separation of duties, high-value review, forecast method/horizon/confidence bounds, forecast-point forecast and period evidence, scenario name/probability/drivers, variance budget/actual/review, collaboration budget/participants, Bytewax routing, supported BFC-agent runtime and role, and human approval for privileged agent actions.

## Configuration

The contract exposes explicit configuration sections:

- `budgets`
- `budget_lines`
- `approvals`
- `forecasts`
- `scenarios`
- `variances`
- `collaboration`
- `bfc_agents`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

Tenant overrides are passed to `get_capability_contract(tenant_id, overrides)` and deep-merged into the default configuration.

## Composition Interfaces

Provides:

- `budget_planning_lifecycle`
- `budget_line_management`
- `budget_approval_workflow`
- `forecast_lifecycle`
- `scenario_planning`
- `variance_analysis_lifecycle`
- `planning_collaboration`
- `bfc_agents`

Requires:

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

## Acceptance Criteria

- `get_capability_contract()` returns a valid APG contract with configuration, schema, deterministic rules, UI routes, theme tokens, and Bytewax streaming metadata.
- Package import exposes `BudgetingForecastingService`, `BFCService`, contract helpers, streaming metadata, and registration metadata without requiring optional web or database dependencies.
- Service supports budget, budget-line, submission, approval, forecast, forecast-point, scenario, variance, collaboration, BFC-agent, dashboard, forecast-summary, variance-summary, audit, batch-validation, and compatibility record operations.
- API helpers and view models expose the same lifecycle surfaces.
- Semantic model includes BFC-agent metadata, required dependencies, route metadata, rules, theme, and Bytewax stream metadata.
- Focused tests cover lifecycle success paths, guardrail failures, API/view execution, app self-test, and semantic metadata.
