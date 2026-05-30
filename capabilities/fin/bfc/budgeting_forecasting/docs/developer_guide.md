# Budgeting and Forecasting Developer Guide

## Public Runtime Surface

Generated APG applications should use the dependency-light package surface:

- `get_capability_contract()`
- `streaming_manifest()`
- `BudgetingForecastingService`
- `BFCService`
- API helpers in `api.py`
- View models in `views.py`
- Semantic and component metadata in `app.py`

The package import path does not require Flask, SQLAlchemy, asyncpg, or a live database.

## Adapter Boundaries

The contract declares adapters for authorization, audit, notifications, general ledger, accounts payable, accounts receivable, cash management, business intelligence, event streaming, and theme. Durable deployments should bind these adapters outside this package surface.

## Event Processing

Lifecycle events use the `apg.fin.bfc.lifecycle` stream and declare `bytewax` as the processor. Durable deployments should attach a Bytewax topology that preserves tenant ordering by `tenant_id`.

## Extending The Capability

When adding lifecycle behavior:

1. Update `SPECIFICATION.md`.
2. Update `PLAN.md` if the implementation path or review gates change.
3. Add or adjust deterministic rules in `capability_contract.py`.
4. Route service writes through `evaluate_capability_rules()`.
5. Add API/view helpers for new routes or workflows.
6. Refresh `semantic_model.json`, `package_manifest.json`, and `release_report.json`.
7. Extend `tests/test_package_contract.py` with success and guardrail coverage.

Do not add live provider imports to `__init__.py`, `api.py`, `views.py`, `service.py`, or `app.py`.
