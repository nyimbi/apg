# Budgeting and Forecasting Deployment Guide

## Package Surface

The package is dependency-light at import time. Generated APG applications can import:

- `get_capability_contract`
- `streaming_manifest`
- `BudgetingForecastingService`
- `BFCService`
- API helpers in `api.py`
- View models in `views.py`
- Semantic and component metadata in `app.py`

## Required Integrations

The contract declares dependencies on:

- Authorization (`auth`)
- Audit (`audl`)
- Notification (`ntfy`)
- Composition events and configuration
- General ledger
- Accounts payable
- Accounts receivable
- Cash management
- Business intelligence

The executable package keeps these as adapter contracts. Durable deployments should bind them to real APG services.

## Readiness Checks

```bash
./.venv/bin/python -m py_compile \
  capabilities/fin/bfc/budgeting_forecasting/__init__.py \
  capabilities/fin/bfc/budgeting_forecasting/capability_contract.py \
  capabilities/fin/bfc/budgeting_forecasting/service.py \
  capabilities/fin/bfc/budgeting_forecasting/api.py \
  capabilities/fin/bfc/budgeting_forecasting/views.py \
  capabilities/fin/bfc/budgeting_forecasting/app.py

./.venv/bin/python capabilities/fin/bfc/budgeting_forecasting/app.py
./.venv/bin/apg capabilities inspect bfc_budgeting_forecasting --json
./.venv/bin/apg capabilities publish-plan capabilities/fin/bfc/budgeting_forecasting --json
```

Do not deploy with a non-Bytewax BFC event processor unless the contract and guardrails are intentionally revised.
