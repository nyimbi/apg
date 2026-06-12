# Cash Management

`cbm_cash_management` is the APG treasury liquidity packet. It owns bank
relationships, cash accounts, cash positions, cash flows, forecasts, liquidity
reviews, bank reconciliation, treasury investments, payment-run funding checks,
and CBM-specific AI agent review lanes.

The public APG boundary is dependency-light. Importing
`capabilities.fin.cbm.cash_management` does not require FastAPI, SQLAlchemy,
Redis, AI providers, bank SDKs, or visualization engines. Optional legacy
helpers remain in the package for deeper integrations, while the contract,
service, API helpers, screen models, and app entrypoint are executable with the
standard APG toolchain.

## What The Capability Provides

- Bank relationship lifecycle.
- Cash account lifecycle by bank, account type, currency, and liquidity buffer.
- Cash position recording with minimum-buffer review controls.
- Cash flow lifecycle for inflows, outflows, and transfers.
- Cash forecast workflow with scenario and confidence controls.
- Liquidity review workflow.
- Bank reconciliation workflow with variance controls.
- Treasury investment workflow with counterparty, maturity, yield, and approval
  controls.
- Payment-run funding control that blocks unapproved projected deficits.
- First-class CBM agents for cash-position, forecast, liquidity,
  reconciliation, investment, and payment-run review.
- UI route metadata and compact theme tokens for composed APG applications.
- Bytewax lifecycle metadata for banks, accounts, positions, flows, forecasts,
  reconciliations, investments, payment runs, and agent events.

## Composition Contract

The executable contract lives in `capability_contract.py`.

Provided capabilities:

- `bank_relationship_lifecycle`
- `cash_account_lifecycle`
- `cash_position_service`
- `cash_flow_lifecycle`
- `cash_forecasting_workflow`
- `liquidity_control_workflow`
- `bank_reconciliation_workflow`
- `treasury_investment_workflow`
- `payment_run_funding_control`
- `cbm_agents`

Required platform capabilities:

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `general_ledger`
- `accounts_payable`
- `accounts_receivable`
- `document_management`
- `business_intelligence`

Lifecycle events are coordinated through Bytewax on
`apg.fin.cbm.lifecycle`.

## Rule Engine

The deterministic rule engine evaluates operation context dictionaries and
returns `allow`, `require_review`, or `deny`. The service calls these rules
before state changes.

Important guardrail families:

- Tenant context and write-policy attachment.
- Bank code and name validation.
- Cash account bank, account number, name, type, and currency validation.
- Cash position account, as-of date, available balance, and liquidity-buffer
  review validation.
- Cash flow account, type, amount, category, and expected-date validation.
- Forecast horizon, scenario, and confidence review validation.
- Reconciliation statement, ledger balance, and variance review validation.
- Treasury investment type, counterparty, maturity, and approval validation.
- Payment-run funding account, current position, and deficit approval
  validation.
- Bytewax-only CBM batch/event routing.
- CBM-agent runtime, role, and privileged-action approval validation.

## Public Python Usage

```python
from capabilities.fin.cbm.cash_management import CashManagementService

service = CashManagementService()
bank = service.create_bank("bank", "tenant-a", "BANK", "Primary Bank")
account = service.create_cash_account(
    "operating",
    "tenant-a",
    bank["id"],
    "001",
    "Operating",
    "operating",
    "USD",
    1000,
)
service.record_cash_position("pos", "tenant-a", account["id"], "2026-05-31", 15000, 15000)
service.record_cash_flow("flow", "tenant-a", account["id"], "outflow", 2500, "supplier", "2026-06-03")
service.create_cash_forecast("forecast", "tenant-a", 30, "base", 0.92)
print(service.dashboard_summary("tenant-a"))
```

## API Helper Usage

`api.py` exposes dependency-light helpers for composition tests and service
adapters:

- `capability_status(tenant_id)`
- `create_bank(payload)`
- `create_cash_account(payload)`
- `record_cash_position(payload)`
- `record_cash_flow(payload)`
- `create_cash_forecast(payload)`
- `record_bank_reconciliation(payload)`
- `create_treasury_investment(payload)`
- `validate_payment_run(payload)`
- `register_cbm_agent(payload)`
- `create_record(payload)`
- `list_records(collection, tenant_id)`

Production adapters should wrap the service with durable stores, bank
connectivity, authorization, audit, notification, document, BI, AP, AR, and GL
dependencies.

## UI And Theming

`views.py` provides screen models for:

- Dashboard
- Banks
- Accounts
- Positions
- Flows
- Forecasts
- Liquidity
- Reconciliation
- Investments
- Payment runs
- Agents
- Settings

The contract exports the `cbm_cash_management_control` theme with compact
treasury-focused tokens and component hints.

## AI Agent Composition

CBM agents are first-class records. Supported runtimes are:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles are:

- `cash_position_reviewer`
- `forecast_reviewer`
- `liquidity_reviewer`
- `bank_reconciliation_reviewer`
- `investment_reviewer`
- `payment_run_reviewer`

Agents may recommend, validate, and prepare work. Privileged actions still
require recorded human approval.

## Verification

Focused package verification:

```bash
./.venv/bin/python -m py_compile \
  capabilities/fin/cbm/cash_management/__init__.py \
  capabilities/fin/cbm/cash_management/capability_contract.py \
  capabilities/fin/cbm/cash_management/service.py \
  capabilities/fin/cbm/cash_management/api.py \
  capabilities/fin/cbm/cash_management/views.py \
  capabilities/fin/cbm/cash_management/app.py \
  capabilities/fin/cbm/cash_management/tests/test_package_contract.py

./.venv/bin/pytest -q capabilities/fin/cbm/cash_management/tests/test_package_contract.py
./.venv/bin/python capabilities/fin/cbm/cash_management/app.py
./.venv/bin/apg capabilities inspect cbm_cash_management --json
./.venv/bin/apg capabilities publish-plan capabilities/fin/cbm/cash_management --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fin/cbm/cash_management --json
```

---

## World-Class Enhancements (v2.0)

- **I1.** Cash Management — World-Class Improvement Catalogue
- **I2.** Probabilistic Cash Flow Forecasting with Confidence Intervals
- **I3.** Real-Time Bank Feed via Open Banking / ISO 20022
- **I4.** Multi-Currency Netting Engine
- **I5.** Liquidity Coverage Ratio (LCR) and Net Stable Funding Ratio (NSFR)
- **I6.** Automated Payment Factory with Priority Queuing
- **I7.** Cash Flow Categorisation via NLP / Pattern Matching
- **I8.** Concentration Risk Monitoring
- **I9.** Automated Bank Reconciliation with Fuzzy Matching
- **I10.** Treasury Investment Optimisation
- **I11.** Working Capital Cycle Analytics
- **I12.** Cash Flow Anomaly Detection
- **I13.** Notional Cash Pooling with Interest Optimisation
- **I14.** SWIFT gpi Payment Tracking Integration
- **I15.** Multi-Entity Consolidated Treasury Dashboard

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
