# Financial Management General Ledger Capability Summary

`glr_general_ledger` provides the executable APG general ledger packet for
chart of accounts, ledger dimensions, periods, journal batches, balanced
journals, postings, reversals, allocations, trial balance, Bytewax lifecycle
events, UI composition, theming, and GLR-agent review lanes.

## Provides

- `chart_of_accounts_lifecycle`
- `ledger_dimension_management`
- `accounting_period_lifecycle`
- `journal_batch_lifecycle`
- `journal_entry_lifecycle`
- `journal_posting_workflow`
- `ledger_balance_service`
- `trial_balance_reporting`
- `allocation_and_reversal_workflow`
- `glr_agents`

## Requires

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `document_management`
- `business_intelligence`
- `financial_reporting`

## Public Runtime

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Screen models: `views.py`
- App entrypoint: `app.py`
- Focused tests: `tests/test_package_contract.py`

## Lifecycle Stream

- Processor: `bytewax`
- Stream: `apg.fin.glr.lifecycle`
- Key: `tenant_id`

## Guardrails

The package enforces tenant context, write policy attachment, account
completeness, period validity, journal source/currency, balanced journal lines,
posting-account validity, exchange-rate presence for foreign currency, posting
approval, open period, idempotency, separation of duties, reversal reason,
trial-balance equality, allocation review, Bytewax routing, GLR-agent runtime
and role validation, and human approval for privileged agent actions.
