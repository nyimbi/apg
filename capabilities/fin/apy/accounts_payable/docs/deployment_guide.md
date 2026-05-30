# Accounts Payable Deployment Guide

## Package Surface

The package is dependency-light at import time. Generated APG applications can import:

- `get_capability_contract`
- `streaming_manifest`
- `AccountsPayableService`
- `APService`
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
- Cash management
- Document management

The executable package keeps these as adapter contracts. Durable deployments should bind them to real APG services.

## Event Processing

Lifecycle events use the `apg.fin.apy.lifecycle` stream and declare `bytewax` as the processor. Durable deployments should attach a Bytewax topology that preserves tenant ordering by `tenant_id`.

## Readiness Checks

```bash
./.venv/bin/python -m py_compile \
  capabilities/fin/apy/accounts_payable/__init__.py \
  capabilities/fin/apy/accounts_payable/capability_contract.py \
  capabilities/fin/apy/accounts_payable/service.py \
  capabilities/fin/apy/accounts_payable/api.py \
  capabilities/fin/apy/accounts_payable/views.py \
  capabilities/fin/apy/accounts_payable/app.py

./.venv/bin/python capabilities/fin/apy/accounts_payable/app.py
./.venv/bin/apg capabilities inspect apy_accounts_payable --json
./.venv/bin/apg capabilities publish-plan capabilities/fin/apy/accounts_payable --json
```

Do not deploy with a non-Bytewax AP event processor unless the contract and guardrails are intentionally revised.
