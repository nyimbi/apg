# Financial Management General Ledger

`glr_general_ledger` is the APG financial system of record. It owns chart of
accounts, ledger dimensions, accounting periods, journal batches, balanced
journal entries, postings, reversals, allocations, trial-balance production,
and GLR-specific AI agent review lanes.

The package is intentionally dependency-light at the APG boundary. Importing
`capabilities.fin.glr.general_ledger` does not require Flask, AppBuilder,
SQLAlchemy sessions, AI providers, or rendering engines. Optional legacy
helpers remain in the package for deeper integrations, while the public
contract, service, API helpers, view models, and app entrypoint are executable
with the standard APG toolchain.

## What The Capability Provides

- Chart-of-accounts lifecycle for active posting accounts and hierarchy-safe
  account setup.
- Ledger dimension capture for department, cost center, project, location, and
  product analysis.
- Accounting period lifecycle with fiscal-year and date controls.
- Journal batch lifecycle for source, currency, and open-period control.
- Journal entry lifecycle with line-count, posting-account, debit/credit, and
  exchange-rate guardrails.
- Posting workflow with approval, idempotency, period, and separation-of-duties
  controls.
- Reversal and allocation workflows with reason, approval, basis, and review
  controls.
- Trial balance reporting that refuses unbalanced ledger evidence.
- First-class GLR agents for journal review, posting review, period close,
  reconciliation, allocation review, and trial-balance review.
- UI route metadata and compact theme tokens for composed APG applications.
- Bytewax lifecycle metadata for accounts, periods, journals, postings,
  reversals, trial balances, allocations, and agent events.

## Composition Contract

The executable contract lives in `capability_contract.py`.

Provided capabilities:

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

Required platform capabilities:

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `document_management`
- `business_intelligence`
- `financial_reporting`

Lifecycle events are coordinated through Bytewax on
`apg.fin.glr.lifecycle`.

## Rule Engine

The deterministic rule engine evaluates operation context dictionaries and
returns `allow`, `require_review`, or `deny`. The service calls these rules
before state changes.

Important guardrail families:

- Tenant context and write-policy attachment.
- Account code, name, type, and hierarchy-cycle validation.
- Period name, fiscal year, date, and range validation.
- Journal batch period, source, and currency validation.
- Journal entry batch, description, line count, posting account, debit/credit,
  and foreign-currency exchange-rate validation.
- Posting approval, open period, idempotency key, and separation-of-duties
  validation.
- Reversal posted-entry and reason validation.
- Trial-balance equality validation.
- Allocation basis and review validation.
- Bytewax-only GLR batch/event routing.
- GLR-agent runtime, role, and privileged-action approval validation.

## Public Python Usage

```python
from capabilities.fin.glr.general_ledger import GeneralLedgerService

service = GeneralLedgerService()
cash = service.create_account("cash", "tenant-a", "1000", "Cash", "asset")
revenue = service.create_account("revenue", "tenant-a", "4000", "Revenue", "revenue")
period = service.open_period("p1", "tenant-a", "FY2026 M01", 2026, "2026-01-01", "2026-01-31")
batch = service.create_journal_batch("b1", "tenant-a", period["id"], "manual")
journal = service.create_journal_entry(
    "j1",
    "tenant-a",
    batch["id"],
    "Record service revenue",
    [
        {"account_id": cash["id"], "debit": 1000, "credit": 0},
        {"account_id": revenue["id"], "debit": 0, "credit": 1000},
    ],
    "preparer",
)
service.approve_journal(journal["id"], "tenant-a", "approver")
service.post_journal(journal["id"], "tenant-a", "poster", "tenant-a:j1:post")
print(service.generate_trial_balance("tenant-a"))
```

## API Helper Usage

`api.py` exposes dependency-light functions for composition tests and service
adapters:

- `capability_status(tenant_id)`
- `create_account(payload)`
- `record_dimension(payload)`
- `open_period(payload)`
- `create_journal_batch(payload)`
- `create_journal_entry(payload)`
- `approve_journal(payload)`
- `post_journal(payload)`
- `reverse_journal(payload)`
- `create_allocation(payload)`
- `register_glr_agent(payload)`
- `create_record(payload)`
- `list_records(collection, tenant_id)`

These helpers use a process-local service instance. Production adapters should
wrap the service with durable storage, authorization, audit, document, BI, and
notification dependencies.

## UI And Theming

`views.py` provides screen models for:

- Dashboard
- Accounts
- Dimensions
- Periods
- Journal batches
- Journals
- Postings
- Trial balance
- Allocations
- Reversals
- Agents
- Settings

The contract exports the `glr_general_ledger_control` theme with compact
ledger-focused tokens and component hints. Composed applications can render the
view models in their own shell while preserving route names, permissions, and
theme semantics.

## AI Agent Composition

GLR agents are first-class records. Supported runtimes are:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles are:

- `journal_reviewer`
- `posting_reviewer`
- `period_close_reviewer`
- `reconciliation_reviewer`
- `allocation_reviewer`
- `trial_balance_reviewer`

Agents may recommend, validate, and prepare work. Privileged actions still
require recorded human approval.

## Verification

Focused package verification:

```bash
./.venv/bin/python -m py_compile \
  capabilities/fin/glr/general_ledger/__init__.py \
  capabilities/fin/glr/general_ledger/capability_contract.py \
  capabilities/fin/glr/general_ledger/service.py \
  capabilities/fin/glr/general_ledger/api.py \
  capabilities/fin/glr/general_ledger/views.py \
  capabilities/fin/glr/general_ledger/app.py \
  capabilities/fin/glr/general_ledger/tests/test_package_contract.py

./.venv/bin/pytest -q capabilities/fin/glr/general_ledger/tests/test_package_contract.py
./.venv/bin/python capabilities/fin/glr/general_ledger/app.py
./.venv/bin/apg capabilities inspect glr_general_ledger --json
./.venv/bin/apg capabilities publish-plan capabilities/fin/glr/general_ledger --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fin/glr/general_ledger --json
```

Full repository tests are intentionally not required for this package-level
slice while working under battery constraints.
