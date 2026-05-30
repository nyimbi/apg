# Cash Management Capability Summary

`cbm_cash_management` provides the executable APG treasury liquidity packet for
bank relationships, cash accounts, positions, flows, forecasts, liquidity
reviews, bank reconciliations, treasury investments, payment-run funding checks,
Bytewax lifecycle events, UI composition, theming, and CBM-agent review lanes.

## Provides

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

## Requires

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

## Public Runtime

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Screen models: `views.py`
- App entrypoint: `app.py`
- Focused tests: `tests/test_package_contract.py`

## Lifecycle Stream

- Processor: `bytewax`
- Stream: `apg.fin.cbm.lifecycle`
- Key: `tenant_id`

## Guardrails

The package enforces tenant context, write policy attachment, bank completeness,
cash-account bank/number/name/type/currency, position account/date/balance,
liquidity-buffer review, cash-flow type/positive amount/category, forecast
horizon/scenario/confidence review, reconciliation statement/ledger/variance
review, investment type/counterparty/maturity/approval, payment-run funding
account/current-position/deficit approval, Bytewax routing, CBM-agent runtime
and role validation, and human approval for privileged agent actions.
